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

struct Task {
    int i, j, task_id;
    float task_cost; 
};
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
inline bool cmp_task_id(const Packet &a, const Packet &b)
{
    return a.task_id < b.task_id;
}
inline MPI_Datatype create_MPI_Packet() {
    MPI_Datatype MPI_Packet;
    int blen[3];
    MPI_Aint array_of_displacements[3];
    MPI_Datatype oldtypes[3];
    blen[0] = 1;
    array_of_displacements[0] = offsetof(Packet, task_penalty);
    oldtypes[0] = MPI_INT;
    blen[1] = 1;
    array_of_displacements[1] = offsetof(Packet, task_id);
    oldtypes[1] = MPI_INT;
    blen[2] = SHA512_STRLEN;
    array_of_displacements[2] = offsetof(Packet, task_hash);
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
    // string answers_hash[total];
    vector<Packet> answers;
    #pragma omp parallel num_threads(2)
    {
        MPI_Status status;
        task_id = 0;
        int task_penalty;
        
        if (omp_get_thread_num() == 0) {
                // load tasks
                queue<Triple> tasks;
                task_id = 0;
                for(int i=1;i<k;i++){
                    for(int j=0;j<i;j++){
                        tasks.push({ i, j, task_id });
                        task_id++;
                    }
                }

            // broadcast initial task
            for (int i = 0; i < size; i++) {
                // send to worker i
                if (tasks.empty()) {
                    // no task
                    Triple task = { NO_MORE_TASK, NO_MORE_TASK, NO_MORE_TASK };
                    MPI_Send(&task, 1, MPI_Triple, i, NEW_TASK_FLAG, comm);
                } else {
                    // new task
                    Triple task = tasks.front();
                    MPI_Send(&task, 1, MPI_Triple, i, NEW_TASK_FLAG, comm);
                    tasks.pop();
                }
            }

            for (int i = 0; i < total; i++) {
                Packet task_result;
                MPI_Recv(&task_result, 1, MPI_Packet, MPI_ANY_SOURCE, COLLECT_RESULT_TAG, comm, &status);
                answers.push_back(task_result);
                // penalties[task_result.task_id] = task_result.task_penalty;
                // answers_hash[task_result.task_id] = task_result.task_hash;
                // cout << "rank[0] from rank[" << status.MPI_SOURCE << "]: task id: " << task_result.task_id << ", penalty: " << task_result.task_penalty << endl;

                // no more task for worker
                if (tasks.empty()) {
                    Triple task = { NO_MORE_TASK, NO_MORE_TASK, NO_MORE_TASK };
                    MPI_Send(&task, 1, MPI_Triple, status.MPI_SOURCE, NEW_TASK_FLAG, comm);
                // more task for worker
                } else {
                    Triple task = tasks.front();
                    MPI_Send(&task, 1, MPI_Triple, status.MPI_SOURCE, NEW_TASK_FLAG, comm);
                    tasks.pop();
                }
                // send new task to worker
                // cout << "rank[0] more task for rank[" << status.MPI_SOURCE << "]: task id:" << i_j_task_id[2] << " (" << i_j_task_id[0] << ", " << i_j_task_id[1] << ") " << endl;
            }

                std::sort(answers.begin(), answers.end(), cmp_task_id);
                
                for (int i = 0; i < total; i++) {

                    // aggregrate answers
                    // alignmentHash = sw::sha512::calculate(alignmentHash.append(answers_hash[i]));
                    alignmentHash = sw::sha512::calculate(alignmentHash.append(answers[i].task_hash));
                    penalties[i] = answers[i].task_penalty;
                }
        } else {
            // uint64_t start, end;
            // start = GetTimeStamp();
            int i, j;
            // initial task
            Triple task;
            MPI_Recv(&task, 1, MPI_Triple, root, NEW_TASK_FLAG, comm, &status);
            i = task.x;
            j = task.y;
            task_id = task.z;

            while (task_id != NO_MORE_TASK) {
                Packet p = do_task(genes[i], genes[j], task_id, pxy, pgap, genes_length[i], genes_length[j]);
                MPI_Send(&p, 1, MPI_Packet, root, COLLECT_RESULT_TAG, comm);
                // cout << "rank[0] done task id: " << p.task_id << ", penalty: " << p.task_penalty << endl;
                MPI_Recv(&task, 1, MPI_Triple, root, NEW_TASK_FLAG, comm, &status);
                i = task.x;
                j = task.y;
                task_id = task.z;
            }

            // end = GetTimeStamp();
            // cout << "rank[" << 0 << "] computes: " <<  end - start  << endl;
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

    
    // uint64_t start, end;
    // start = GetTimeStamp();
    // worker works
    MPI_Datatype MPI_Packet = create_MPI_Packet();
    MPI_Datatype MPI_Triple = create_MPI_Triple();
    MPI_Status status;
    int i, j, task_id;
    // initial task
    Triple task;
    MPI_Recv(&task, 1, MPI_Triple, root, NEW_TASK_FLAG, comm, &status);
    i = task.x;
    j = task.y;
    task_id = task.z;

    while (task_id != NO_MORE_TASK) {
        Packet p = do_task(genes[i], genes[j], task_id, pxy, pgap, genes_length[i], genes_length[j]);
        MPI_Send(&p, 1, MPI_Packet, root, COLLECT_RESULT_TAG, comm);
        // cout << "rank[0] done task id: " << p.task_id << ", penalty: " << p.task_penalty << endl;
        MPI_Recv(&task, 1, MPI_Triple, root, NEW_TASK_FLAG, comm, &status);
        i = task.x;
        j = task.y;
        task_id = task.z;
    }

    // end = GetTimeStamp();
    // cout << "rank[" << rank << "] computes: " <<  end - start  << endl;
}

inline int getMinimumPenalty2(std::string x, std::string y, int pxy, int pgap, int *xans, int *yans, int m, int n) {
	int i, j; // intialising variables
    omp_set_num_threads(n_threads);

    // int m = x.length(); // length of gene1
    // int n = y.length(); // length of gene2
    int row = m + 1, col = n + 1;

    // table for storing optimal substructure answers
    int **dp = new2d(row, col);
    //	size_t size = m + 1;
    //	size *= n + 1;
    //	memset (dp[0], 0, size);

    // intialising the table
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i * pgap;
        }
        #pragma omp for
        for (int i = 1; i <= n; i++) {
            dp[0][i] = i * pgap;
        }
    }

    // calcuting the minimum penalty
    
    // calculate tile size
    int tile_width  = (int) ceil((1.0*m) / n_threads), 
        tile_length = (int) ceil((1.0*n) / n_threads);
    int num_tile_in_width = (int) ceil((1.0*m) / tile_width);
    int num_tile_in_length = (int) ceil((1.0*n) / tile_length);

    // modified from: https://www.geeksforgeeks.org/zigzag-or-diagonal-traversal-of-matrix/
    // There will be tile_width + num_tile_in_length-1 lines in the output
    for (int line = 1; line <= (num_tile_in_width + num_tile_in_length - 1); line++) {
        /* Get column index of the first element in this line of output.
           The index is 0 for first tile_width lines and line - tile_width for remaining
           lines  */
        int start_col = max(0, line - num_tile_in_width);

        /* Get count of elements in this line. The count of elements is
           equal to minimum of line number, num_tile_in_length-start_col and num_tile_in_width */
        int count = min(line, min((num_tile_in_length - start_col), num_tile_in_width));

        // parallel each tile on anti-diagonal
        #pragma omp parallel for
        for (int z = 0; z < count; z++) {
            int tile_i_start = (min(num_tile_in_width, line)-z-1)*tile_width +1,
                tile_j_start = (start_col+z)*tile_length +1;

            // sequential calculate cells in tile
            for (int i = tile_i_start; i < min(tile_i_start + tile_width, row); i++) {
                for (int j = tile_j_start; j < min(tile_j_start + tile_length, col); j++) {

                    if (x[i - 1] == y[j - 1]) {
                        dp[i][j] = dp[i - 1][j - 1];
                    } else {
                        dp[i][j] = min3(dp[i - 1][j - 1] + pxy ,
                                dp[i - 1][j] + pgap ,
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
            xans[xpos--] = (int)x[i - 1];
            yans[ypos--] = (int)y[j - 1];
            i--;
            j--;
        } else if (dp[i - 1][j - 1] + pxy == dp[i][j]) {
            xans[xpos--] = (int)x[i - 1];
            yans[ypos--] = (int)y[j - 1];
            i--;
            j--;
        } else if (dp[i - 1][j] + pgap == dp[i][j]) {
            xans[xpos--] = (int)x[i - 1];
            yans[ypos--] = (int)'_';
            i--;
        } else {
            xans[xpos--] = (int)'_';
            yans[ypos--] = (int)y[j - 1];
            j--;
        }
    }
    
    // while (xpos > 0) {
	// 	if (i > 0) xans[xpos--] = (int)x[--i];
	// 	else xans[xpos--] = (int)'_';
	// }
	// while (ypos > 0) {
	// 	if (j > 0) yans[ypos--] = (int)y[--j];
	// 	else yans[ypos--] = (int)'_';
	// }

    int x_diff = xpos - i, y_diff = ypos - j;
    #pragma omp parallel for
    for (int ii = i; ii>0; ii--){
        xans[ii + x_diff] = (int)x[ii - 1];
    }
    #pragma omp parallel for
    for (int x_dash = x_diff; x_dash>0; x_dash--){
        xans[x_dash] = (int)'_';
    }
    #pragma omp parallel for
    for (int jj = j; jj>0; jj--){
        yans[jj + y_diff] = (int)y[jj - 1];
    }
    #pragma omp parallel for
    for (int y_dash = y_diff; y_dash>0; y_dash--){
        yans[y_dash] = (int)'_';
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
