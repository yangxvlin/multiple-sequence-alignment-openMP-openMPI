// CPP program to solve the sequence alignment
// problem. Adapted from https://www.geeksforgeeks.org/sequence-alignment-problem/ 
// with many fixes and changes for multiple sequence alignment and to include an MPI driver
#include <mpi.h>
#include <sys/time.h>
#include <string>
#include <cstring>
#include <iostream>
#include "sha512.hh"

using namespace std;

std::string getMinimumPenalties(std::string *genes, int k, int pxy, int pgap, int *penalties);
int getMinimumPenalty(std::string x, std::string y, int pxy, int pgap, int *xans, int *yans);
void do_MPI_task(int rank);

/*
Examples of sha512 which returns a std::string
sw::sha512::calculate("SHA512 of std::string") // hash of a string, or
sw::sha512::file(path) // hash of a file specified by its path, or
sw::sha512::calculate(&data, sizeof(data)) // hash of any block of data
*/

// Return current time, for performance measurement
uint64_t GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
}

const MPI_Comm comm = MPI_COMM_WORLD;
const int root = 0;

// Driver code
int main(int argc, char **argv){
	int rank;
	int prov;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &prov);
	MPI_Comm_rank(comm, &rank);
	if(rank==root){
		int misMatchPenalty;
		int gapPenalty;
		int k;
		std::cin >> misMatchPenalty;
		std::cin >> gapPenalty;
		std::cin >> k;	
		std::string genes[k];
		for(int i=0;i<k;i++) std::cin >> genes[i];

		int numPairs= k*(k-1)/2;

		int penalties[numPairs];
		
		uint64_t start = GetTimeStamp ();

		// return all the penalties and the hash of all allignments
		std::string alignmentHash = getMinimumPenalties(genes,
			k,misMatchPenalty, gapPenalty,
			penalties);
		
		// print the time taken to do the computation
		printf("Time: %ld us\n", (uint64_t) (GetTimeStamp() - start));
		
		// print the alginment hash
		std::cout<<alignmentHash<<std::endl;

		for(int i=0;i<numPairs;i++){
			std::cout<<penalties[i] << " ";
		}
		std::cout << std::endl;
	} else {
		// do stuff for MPI tasks that are not rank==root
		do_MPI_task(rank);
	}
	MPI_Finalize();
	return 0;
}

/******************************************************************************/
/* Do not change any lines above here.            */
/* All of your changes should be below this line. */
/******************************************************************************/
#include <omp.h>
#include <math.h>
#include <set>
#include <unordered_set>
#include <vector>
#include <queue> 
#include <map>
#include <algorithm>    // std::sort

// uncomment to enable debug
// #define debug 0

int getMinimumPenalty2(std::string x, std::string y, int pxy, int pgap, int *xans, int *yans, int m, int n);

int n_threads = 16;
constexpr int SHA512_STRLEN = 129; // 128+1 for '\0'

constexpr int NO_MORE_TASK = -1;
constexpr int ask_for_genes_tag = 1;
constexpr int send_genes_tag = 2;
constexpr int COLLECT_RESULT_TAG = 3;
constexpr int COLLECT_RESULT_TAG2 = 4;
constexpr int COLLECT_RESULT_TAG3 = 5;
constexpr int NEW_TASK_FLAG = 6;

struct Triple { 
   int x, y, z; 
}; 
inline MPI_Datatype create_MPI_Triple()
{
    // define the Triple type for MPI
    MPI_Datatype MPI_Triple;
    MPI_Type_contiguous(3, MPI_INT, &MPI_Triple);
    MPI_Type_commit(&MPI_Triple);
    return MPI_Triple;
}

// struct Task {
//     int i, j, task_id;
//     float task_cost; 
// };
// struct task_cost_cmp {
//     bool operator()(const Task & a, const Task & b) {
//         // largest comes first
//         return a.task_cost < b.task_cost;
//     }
// };

// struct Quatic {
//     int x, y, z, r; 
// };

struct Packet {
    int task_penalty;
    int task_id;
    char task_hash[SHA512_STRLEN];
};
// inline bool cmp_task_id(const Packet &a, const Packet &b)
// {
//     return a.task_id < b.task_id;
// }
inline MPI_Datatype create_MPI_Packet() {
    MPI_Datatype MPI_Packet;
    int blen[3];
    blen[0] = 1;
    blen[1] = 1;
    blen[2] = SHA512_STRLEN;
    MPI_Aint array_of_displacements[3];
    array_of_displacements[0] = offsetof(Packet, task_penalty);
    array_of_displacements[1] = offsetof(Packet, task_id);
    array_of_displacements[2] = offsetof(Packet, task_hash);
    MPI_Datatype oldtypes[3];
    oldtypes[0] = MPI_INT;
    oldtypes[1] = MPI_INT;
    oldtypes[2] = MPI_CHAR;
    MPI_Type_create_struct(3, blen, array_of_displacements, oldtypes, &MPI_Packet);
    MPI_Type_commit(&MPI_Packet);
    return MPI_Packet;
}

inline int min3(int a, int b, int c) {
	if (a <= b && a <= c) {
		return a;
	} else if (b <= a && b <= c) {
		return b;
	} else {
		return c;
	}
}

// equivalent of  int *dp[width] = new int[height][width]
// but works for width not known at compile time.
// (Delete structure by  delete[] dp[0]; delete[] dp;)
inline int **new2d (int width, int height) {
	int **dp = new int *[width];
	size_t size = width;
	size *= height;
	int *dp0 = new int [size];
	if (!dp || !dp0) {
	    std::cerr << "getMinimumPenalty: new failed" << std::endl;
	    exit(1);
	}
	dp[0] = dp0;
	for (int i = 1; i < width; i++)
	    dp[i] = dp[i-1] + height;

	return dp;
}

inline Packet do_task(std::string gene1, std::string gene2, 
                      int task_id, int pxy, int pgap, 
                      int m, int n) {

    int l = m + n;
    int xans[l + 1], yans[l + 1];
    int penalty = getMinimumPenalty2(gene1, gene2, pxy, pgap, xans, yans, m, n);

    int id = 1;
    int a;

    // find the start of the extra gap
    for (a = l; a >= 1; a--)
    {
        if ((char)yans[a] == '_' && (char)xans[a] == '_')
        {
            id = a + 1;
            break;
        }
    }

    // extract the exact alignment for both string
    std::string align1 = "";
    std::string align2 = "";
    for (a = id; a <= l; a++)
    {
        align1.append(1, (char)xans[a]);
    }
    for (a = id; a <= l; a++)
    {
        align2.append(1, (char)yans[a]);
    }

    // alignmentHash = hash(alignmentHash ++ hash(hash(align1)++hash(align2)))
    std::string align1hash = sw::sha512::calculate(align1);
    std::string align2hash = sw::sha512::calculate(align2);
    std::string problemhash = sw::sha512::calculate(align1hash.append(align2hash));

    Packet p;
    p.task_penalty = penalty;
    p.task_id = task_id;
    strcpy(p.task_hash, problemhash.c_str());

    return p;
}

// called by the root MPI task only
// this procedure should distribute work to other MPI tasks
// and put together results, etc.
inline std::string getMinimumPenalties(std::string *genes, 
                                       int k, 
                                       int pxy, 
                                       int pgap,
	                                   int *penalties) {
    std::string alignmentHash="";
    n_threads--;

    omp_set_nested(1);  /* make sure nested parallism is on */

    int task_id;
    // number of processes
    int size;
    MPI_Comm_size(comm, &size);

    // broadcast k, pxy, pgap to wrokers
    int k_pxy_pgap[3] = {k, pxy, pgap};
    MPI_Bcast(k_pxy_pgap, 3, MPI_INT, root, comm);

    // total tasks
    int total = k * (k-1) / 2;
    // calculates string length
    int genes_length[k];
    for (int i = 0; i < k; i++) {
        genes_length[i] = genes[i].length();
    }
    // broadcast strings length to wrokers
    MPI_Bcast(genes_length, k, MPI_INT, root, comm);

    // broadcast strings to wrokers
    // int max_gene_len = *std::max_element(genes_length, genes_length + k) + 1;
    for (int i = 0; i < k; i++) {
        char str_buffer[genes_length[i]];
        memcpy(str_buffer, genes[i].c_str(), genes_length[i]);
        MPI_Bcast(str_buffer, genes_length[i], MPI_CHAR, root, comm);
    }

    MPI_Datatype MPI_Packet = create_MPI_Packet();
    MPI_Datatype MPI_Triple = create_MPI_Triple();

    // cout << "111111" << endl;
    // master's dynamic task control
    #pragma omp parallel num_threads(2)
    {
        MPI_Status status;
        task_id = 0;
        // int task_penalty;
        
        if (omp_get_thread_num() == 0) {
            // load tasks
            vector<Triple> tasks;
            task_id = 0;
            for(int i=1;i<k;i++){
                for(int j=0;j<i;j++){
                    tasks.push_back({ i, j, task_id });
                    task_id++;
                }
            }

            // using built-in random generator:
            std::random_shuffle( tasks.begin(), tasks.end() );

            // broadcast initial task
            for (int i = 0; i < size; i++) {
            // for (int i = 1; i < size; i++) {
                // send to worker i
                if (tasks.empty()) {
                    // no task
                    Triple task = { NO_MORE_TASK, NO_MORE_TASK, NO_MORE_TASK };
                    MPI_Send(&task, 1, MPI_Triple, i, NEW_TASK_FLAG, comm);
                } else {
                    // new task
                    Triple task = tasks.back();
                    MPI_Send(&task, 1, MPI_Triple, i, NEW_TASK_FLAG, comm);
                    tasks.pop_back();
                }
            }

            Packet answers[total];
            // vector<Packet> answers;
            for (int i = 0; i < total; i++) {
                Packet task_result;
                MPI_Recv(&task_result, 1, MPI_Packet, MPI_ANY_SOURCE, COLLECT_RESULT_TAG, comm, &status);
                answers[task_result.task_id] = task_result;
                // answers.push_back(task_result);
                // penalties[task_result.task_id] = task_result.task_penalty;
                // answers_hash[task_result.task_id] = task_result.task_hash;
                // cout << "rank[0] from rank[" << status.MPI_SOURCE << "]: task id: " << task_result.task_id << ", penalty: " << task_result.task_penalty << endl;

                // no more task for worker
                if (tasks.empty()) {
                    Triple task = { NO_MORE_TASK, NO_MORE_TASK, NO_MORE_TASK };
                    MPI_Send(&task, 1, MPI_Triple, status.MPI_SOURCE, NEW_TASK_FLAG, comm);
                // more task for worker
                } else {
                    Triple task = tasks.back();
                    MPI_Send(&task, 1, MPI_Triple, i, NEW_TASK_FLAG, comm);
                    tasks.pop_back();
                }
                // send new task to worker
                // cout << "rank[0] more task for rank[" << status.MPI_SOURCE << "]: task id:" << i_j_task_id[2] << " (" << i_j_task_id[0] << ", " << i_j_task_id[1] << ") " << endl;
            }
            
            // std::sort(answers.begin(), answers.end(), cmp_task_id);
            // for (int i = 0; i < total; i++) {
            //     // aggregrate answers
            //     // alignmentHash = sw::sha512::calculate(alignmentHash.append(answers_hash[i]));
            //     alignmentHash = sw::sha512::calculate(alignmentHash.append(answers[i].task_hash));
            //     // penalties[i] = answers[i].task_penalty;
            // }
            
            // aggregrate answer hashs
            for (int i = 0; i < total; i++) {
                penalties[i] = answers[i].task_penalty;
                alignmentHash = sw::sha512::calculate(alignmentHash.append(answers[i].task_hash));
            }

        } else {
            uint64_t start, end, start1, end1;
            start = GetTimeStamp();
            Triple task;
            do {
                MPI_Recv(&task, 1, MPI_Triple, root, NEW_TASK_FLAG, comm, &status);
                if (task.z == NO_MORE_TASK) {
                    break;
                }
                // start1 = GetTimeStamp();
                Packet p = do_task(genes[task.x], genes[task.y], task.z, pxy, pgap, genes_length[task.x], genes_length[task.y]);
                MPI_Send(&p, 1, MPI_Packet, root, COLLECT_RESULT_TAG, comm);
                // end1 = GetTimeStamp();
                cout << "rank[" << 0 << "] computes: " <<  end1 - start1 << " for task: " << task.z << " with length: " << 
                genes_length[task.x] << ", " << genes_length[task.y] << endl;
            } while (true);

            end = GetTimeStamp();
            cout << "rank[" << 0 << "] computes: " <<  end - start  << endl;
        }
    }
    // cout << "77777777" << endl;

	return alignmentHash;
}


// called for all tasks with rank!=root
// do stuff for each MPI task based on rank
inline void do_MPI_task(int rank) {
    
    // number of processes
    int size;
    MPI_Comm_size(comm, &size);

    // broadcast strings length from master
    int k_pxy_pgap[3];
    MPI_Bcast(k_pxy_pgap, 3, MPI_INT, root, comm);
    int k = k_pxy_pgap[0], 
        pxy = k_pxy_pgap[1],
        pgap = k_pxy_pgap[2];

    // total tasks
    int total = k * (k-1) / 2;

    // broadcast strings length from master
    int genes_length[k];
    MPI_Bcast(genes_length, k, MPI_INT, root, comm);
    
    // broadcast strings from master
    // int max_gene_len = *std::max_element(genes_length, genes_length + k) + 1;
    string genes[k];
    for (int i = 0; i < k; i++) {
        char buffer[genes_length[i] + 1];
        MPI_Bcast(buffer, genes_length[i], MPI_CHAR, root, comm);
        buffer[genes_length[i]] = '\0';
        genes[i] = string(buffer, genes_length[i]);
    }

    
    uint64_t start, end;
    start = GetTimeStamp();
    // worker works
    MPI_Datatype MPI_Packet = create_MPI_Packet();
    MPI_Datatype MPI_Triple = create_MPI_Triple();
    MPI_Status status;
    Triple task;
    do {
        MPI_Recv(&task, 1, MPI_Triple, root, NEW_TASK_FLAG, comm, &status);
        if (task.z == NO_MORE_TASK) {
            break;
        }
        Packet p = do_task(genes[task.x], genes[task.y], task.z, pxy, pgap, genes_length[task.x], genes_length[task.y]);
        MPI_Send(&p, 1, MPI_Packet, root, COLLECT_RESULT_TAG, comm);
    } while (true);

    end = GetTimeStamp();
    cout << "rank[" << rank << "] computes: " <<  end - start  << endl;
}

inline int getMinimumPenalty2(std::string x, std::string y, int pxy, int pgap, int *xans, int *yans, int m, int n) {
	int i, j;

    // int m = x.length(); // length of gene1
    // int n = y.length(); // length of gene2

    omp_set_num_threads(n_threads);
    int row = m + 1;
    int col = n + 1;
    int **dp = new2d(row, col);

    // size_t size = row;
    // size *= col;
    // memset (dp[0], 0, size);

    // intialising the table
    #pragma omp parallel
    {
        // if (omp_get_thread_num() == 0 && omp_get_num_threads() <= 15) {
        //     cout << omp_get_num_threads() <<" threads, n_threads=" << n_threads << endl;
        // }

        #pragma omp for nowait
        for (i = 0; i <= m; i++) {
            dp[i][0] = i * pgap;
        }
        #pragma omp for
        for (i = 0; i <= n; i++) {
            dp[0][i] = i * pgap;
        }
    }

    // calculate tile size
    int tile_width = (int)ceil((1.0 * m) / n_threads); 
    int tile_length = (int)ceil((1.0 * n) / n_threads);
    int num_tile_in_width = (int)ceil((1.0 * m) / tile_width);
    int num_tile_in_length = (int)ceil((1.0 * n) / tile_length);

    // modified from: https://www.geeksforgeeks.org/zigzag-or-diagonal-traversal-of-matrix/
    // There will be tile_width + num_tile_in_length-1 lines in the output
    for (int line = 1; line <= (num_tile_in_width + num_tile_in_length - 1); line++) {
        /* Get column index of the first element in this line of output. */
        int start_col = max(0, line - num_tile_in_width) + 1;
        /* Get count of elements in this line.  */
        int count = min(line, num_tile_in_width);

        #pragma omp parallel for
        for (int z = start_col; z <= count; z++) {
            int tile_i_start = (z - 1) * tile_width + 1;              
            int tile_i_end = min(tile_i_start + tile_width, row); 
            int tile_j_start = (line - z) * tile_length + 1; 
            int tile_j_end = min(tile_j_start + tile_length, col); 

            // sequential calculate cells in tile
            for (int i = tile_i_start; i < tile_i_end; i++) {
                for (int j = tile_j_start; j < tile_j_end; j++) {
                    if (x[i - 1] == y[j - 1]) {
                        dp[i][j] = dp[i - 1][j - 1];
                    } else {
                        // dp[i][j] = min3(dp[i - 1][j - 1] + pxy,
                        //                 dp[i - 1][j] + pgap,
                        //                 dp[i][j - 1] + pgap);
                        dp[i][j] = min(min(dp[i - 1][j - 1] + pxy,
                                        dp[i - 1][j] + pgap),
                                        dp[i][j - 1] + pgap);
                    }
                }
            }
        }
    }

    // Reconstructing the solution
    int l = n + m; // maximum possible length

    i = m;
    j = n;

    int xpos = l;
    int ypos = l;

    while (!(i == 0 || j == 0)) {
        if (x[i - 1] == y[j - 1]) {
            xans[xpos--] = (int) x[--i];
            yans[ypos--] = (int) y[--j];
            // xans[xpos--] = (int) x[i - 1];
            // yans[ypos--] = (int) y[j - 1];
            // i--;
            // j--;
        } else if (dp[i - 1][j - 1] + pxy == dp[i][j]) {
            xans[xpos--] = (int) x[--i];
            yans[ypos--] = (int) y[--j];
            // xans[xpos--] = (int) x[i - 1];
            // yans[ypos--] = (int) y[j - 1];
            // i--;
            // j--;
        } else if (dp[i - 1][j] + pgap == dp[i][j]) {
            xans[xpos--] = (int) x[--i];
            yans[ypos--] = (int) '_';
            // xans[xpos--] = (int) x[i - 1];
            // yans[ypos--] = (int) '_';
            // i--;
        // } else if (dp[i][j - 1] + pgap == dp[i][j]) {
        } else {
            xans[xpos--] = (int) '_';
            yans[ypos--] = (int) y[--j];
            // xans[xpos--] = (int) '_';
            // yans[ypos--] = (int) y[j - 1];
            // j--;
        }
    }

    int x_offset = xpos - i, y_offset = ypos - j;
    omp_set_num_threads(n_threads);
    #pragma omp parallel 
    {
        #pragma omp for nowait
        for (int ii = i; ii > 0; ii--)
        {
            xans[ii + x_offset] = (int)x[ii - 1];
        }

        #pragma omp for nowait
        for (int x_pos2 = x_offset; x_pos2 > 0; x_pos2--) {
            xans[x_pos2] = (int)'_';
        }

        #pragma omp for nowait
        for (int jj = j; jj > 0; jj--) {
            yans[jj + y_offset] = (int)y[jj - 1];
        }

        #pragma omp for nowait
        for (int y_pos2 = y_offset; y_pos2 > 0; y_pos2--) {
            yans[y_pos2] = (int)'_';
        }
    }

    int ret = dp[m][n];

    delete[] dp[0];
    delete[] dp;

    return ret;
}

// function to find out the minimum penalty
// return the minimum penalty and put the aligned sequences in xans and yans
int getMinimumPenalty(std::string x, std::string y, int pxy, int pgap, int *xans, int *yans) {
	
	int i, j; // intialising variables

	int m = x.length(); // length of gene1
	int n = y.length(); // length of gene2
	
	// table for storing optimal substructure answers
	int **dp = new2d (m+1, n+1);
	size_t size = m + 1;
	size *= n + 1;
	memset (dp[0], 0, size);

	// intialising the table
	for (i = 0; i <= m; i++)
	{
		dp[i][0] = i * pgap;
	}
	for (i = 0; i <= n; i++)
	{
		dp[0][i] = i * pgap;
	}

	// calcuting the minimum penalty
	for (i = 1; i <= m; i++)
	{
		for (j = 1; j <= n; j++)
		{
			if (x[i - 1] == y[j - 1])
			{
				dp[i][j] = dp[i - 1][j - 1];
			}
			else
			{
				dp[i][j] = min3(dp[i - 1][j - 1] + pxy ,
						dp[i - 1][j] + pgap ,
						dp[i][j - 1] + pgap);
			}
		}
	}

	// Reconstructing the solution
	int l = n + m; // maximum possible length
	
	i = m; j = n;
	
	int xpos = l;
	int ypos = l;
	
	while ( !(i == 0 || j == 0))
	{
		if (x[i - 1] == y[j - 1])
		{
			xans[xpos--] = (int)x[i - 1];
			yans[ypos--] = (int)y[j - 1];
			i--; j--;
		}
		else if (dp[i - 1][j - 1] + pxy == dp[i][j])
		{
			xans[xpos--] = (int)x[i - 1];
			yans[ypos--] = (int)y[j - 1];
			i--; j--;
		}
		else if (dp[i - 1][j] + pgap == dp[i][j])
		{
			xans[xpos--] = (int)x[i - 1];
			yans[ypos--] = (int)'_';
			i--;
		}
		else if (dp[i][j - 1] + pgap == dp[i][j])
		{
			xans[xpos--] = (int)'_';
			yans[ypos--] = (int)y[j - 1];
			j--;
		}
	}
	while (xpos > 0)
	{
		if (i > 0) xans[xpos--] = (int)x[--i];
		else xans[xpos--] = (int)'_';
	}
	while (ypos > 0)
	{
		if (j > 0) yans[ypos--] = (int)y[--j];
		else yans[ypos--] = (int)'_';
	}

	int ret = dp[m][n];

	delete[] dp[0];
	delete[] dp;
	
	return ret;
}
