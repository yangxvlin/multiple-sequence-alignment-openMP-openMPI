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
	MPI_Init(&argc, &argv);
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
#include <algorithm>    // std::sort

// uncomment to enable debug
// #define debug 0

int getMinimumPenalty2(std::string x, std::string y, int pxy, int pgap, int *xans, int *yans, int m, int n, int **dp);

// const int n_threads = 16;
const int sha512_strlen = 128 + 1; // +1 for '\0'

const int ask_for_genes_tag = 1;
const int send_genes_tag = 2;
const int collect_results_tag = 3;
const int collect_results_tag2 = 4;
const int collect_results_tag3 = 5;

struct Triple { 
   int x, y, z; 
}; 

struct Quatic {
    int x, y, z, r; 
};

struct Packet {
    int task_id;
    int task_penalty;
    char task_hash[sha512_strlen];
};

int min3(int a, int b, int c) {
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
int **new2d (int width, int height) {
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

// called by the root MPI task only
// this procedure should distribute work to other MPI tasks
// and put together results, etc.
std::string getMinimumPenalties(std::string *genes, 
                                       int k, 
                                       int pxy, 
                                       int pgap,
	                                   int *penalties) {
    uint64_t start = GetTimeStamp();

    MPI_Status status;
    int n_threads = omp_get_max_threads();
    // cout << "rank[" << 0 << "] has threads: " << n_threads << endl;
    omp_set_num_threads(n_threads);
	int probNum=0;

    int size;
    MPI_Comm_size(comm, &size);

    // send k, pxy, pgap to wrokers
    int k_pxy_pgap[3] = {k, pxy, pgap};
    MPI_Bcast(k_pxy_pgap, 3, MPI_INT, root, comm);

    int total = k * (k-1) / 2;
    // calculates string length
    int genes_length[k];
    for (int i = 0; i < k; i++) {
        genes_length[i] = genes[i].length();
    }
    MPI_Bcast(genes_length, k, MPI_INT, root, comm);

    int max_gene_len = *std::max_element(genes_length, genes_length + k) + 1;
    // #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < k; i++) {
        char buffer[max_gene_len];
        memcpy(buffer, genes[i].c_str(), genes_length[i]);
        MPI_Bcast(buffer, genes_length[i], MPI_CHAR, root, comm);
    }

    #ifdef DEBUG
        cout << "rank[0][tasks generation start]" << endl;
    #endif // DEBUG

    // do root's tasks
    // number of dp matrix calculation per process
    // int tasks_per_process = (int) floor((1.0*total) / size);
    // int my_tasks_start = tasks_per_process * (size-1), my_tasks_end = total; // lask chunk of tasks on root

    uint64_t end = GetTimeStamp();
    cout << "11111111 " << end - start << endl;
    start = GetTimeStamp();

    // ask root for the genes needed for calculation
    int task_id = 0;
    // calculate calculation cells in each task and distribute evenly
    unsigned long long n_cells[total];
    unsigned long long total_cells = 0;
    for(int i=1;i<k;i++){
		for(int j=0;j<i;j++){
            // cout << genes_length[i] << " " << genes_length[j] << endl;
            n_cells[task_id] = (long) genes_length[i] * genes_length[j];
            total_cells += n_cells[task_id];
            // cout << i << " " << j << " " << n_cells[task_id] << " " << total_cells <<endl;
            task_id++;
        }
    }

    end = GetTimeStamp();
    cout << "22222222 " << end - start << endl;
    start = GetTimeStamp();

    unsigned long long cells_per_proccess = total_cells / size;
    vector<Triple> tasks[size]; // i, j, id of (i, j) in whole tasks
    task_id = 0;
    // int task_rank_mapping[total];
    unsigned long long rank_load[size];
    for (int i = 0; i < size; i++) {
        rank_load[i] = 0;
    }
    for(int i=1;i<k;i++){
		for(int j=0;j<i;j++){
            // make rank 0 as minimum task load as possible
            for (int r = size-1; r >= 0; r-- ) {
                // load task to rank
                if (rank_load[r] + n_cells[task_id] <= cells_per_proccess) {
                    rank_load[r] += n_cells[task_id];
                    tasks[r].push_back({ i, j, task_id });
                    break;
                }
            }
            task_id++;
        }
    }

    end = GetTimeStamp();
    cout << "33333333 " << end - start << endl;
    start = GetTimeStamp();

    int i_max_length = -1, j_max_length = -1;
    for (int z = 0; z < tasks[0].size(); z++) {
        Triple t = tasks[0].at(z);
        if (genes_length[t.x] > i_max_length) {
            i_max_length = genes_length[t.x];
        }
        if (genes_length[t.y] > j_max_length) {
            j_max_length = genes_length[t.y];
        }
    }

    end = GetTimeStamp();
    cout << "44444444 " << end - start << endl;
    start = GetTimeStamp();

    // create dp matrix for all tasks
    int **dp = new2d(i_max_length + 1, j_max_length + 1);
    // intialising the table
    #pragma omp parallel 
    {
        #pragma omp for nowait
        for (int i = 0; i <= i_max_length; i++) {
            dp[i][0] = i * pgap;
        }
        #pragma omp for
        for (int i = 1; i <= j_max_length; i++) {
            dp[0][i] = i * pgap;
        }
    }

    #ifdef DEBUG
        cout << "rank[0][tasks generation finish]" << endl;
    #endif // DEBUG

    end = GetTimeStamp();
    cout << "55555555 " << end - start << endl;
    start = GetTimeStamp();

    string answers_hash[total];
    // char answers_hash_array[total][sha512_strlen];
    int n_tasks = tasks[0].size();
    for (int z = 0; z < n_tasks; z++) {
        // cout << "rank[0] " << z << endl;
        Triple xyz = tasks[0].at(z);
        int i = xyz.x, j = xyz.y, task_id = xyz.z;
        int l = genes_length[i] + genes_length[j];
        int xans[l+1], yans[l+1];
        penalties[task_id] = getMinimumPenalty2(genes[i], genes[j], pxy, pgap, xans, yans, genes_length[i], genes_length[j], dp);
        // Since we have assumed the answer to be n+m long,
        // we need to remove the extra gaps in the starting
        // id represents the index from which the arrays
        // xans, yans are useful
        int id = 1;
        int a;
        for (a = l; a >= 1; a--) {
            if ((char)yans[a] == '_' && (char)xans[a] == '_') {
                id = a + 1;
                break;
            }
        }
        std::string align1="";
        std::string align2="";
        for (a = id; a <= l; a++) {
            align1.append(1,(char)xans[a]);
        }
        for (a = id; a <= l; a++) {
            align2.append(1,(char)yans[a]);
        }
        std::string align1hash = sw::sha512::calculate(align1);
        std::string align2hash = sw::sha512::calculate(align2);
        std::string problemhash = sw::sha512::calculate(align1hash.append(align2hash));

        // store problemhash to root
        answers_hash[task_id] = problemhash;
        #ifdef DEBUG
            cout << "rank[0][calc] " << "task id: " << task_id << ", penalty: " << penalties[task_id] << ", hash: " << answers_hash[task_id] << endl;
        #endif // DEBUG
    }

    end = GetTimeStamp();
    cout << "66666666 rank[0]" << end - start << endl;
    start = GetTimeStamp();

    #ifdef DEBUG
        cout << "rank[0][calc] finish" << endl;
    #endif // DEBUG
    // recv results form worker
    
    // #pragma omp parallel for schedule(static, 1)
    for (int i = 1; i < size; i++) {
        int rank_task_size;
        MPI_Recv(&rank_task_size, 1, MPI_INT, MPI_ANY_SOURCE, collect_results_tag, comm, &status);
        int task_penalties[rank_task_size];
        MPI_Recv(task_penalties, rank_task_size, MPI_INT, status.MPI_SOURCE, collect_results_tag2, comm, &status);
        // MPI_Recv(task_penalties, rank_task_size, MPI_INT, i, collect_results_tag2, comm, &status);
        char buffer[sha512_strlen];
        
        for (int j = 0; j < rank_task_size; j++) {
            int task_id = tasks[status.MPI_SOURCE].at(j).z;
            penalties[task_id] = task_penalties[j];

            MPI_Recv(buffer, 128, MPI_CHAR, status.MPI_SOURCE, collect_results_tag3, comm, &status);
            answers_hash[task_id] = string(buffer, 128);
            #ifdef DEBUG
                cout << "id: " << task_id << ", " << answers_hash[task_id] << endl;
            #endif // DEBUG
        }
        #ifdef DEBUG
            cout << "rank[0][recv] answer from " << "rank: " << i << endl;
        #endif // DEBUG
    }

    end = GetTimeStamp();
    cout << "77777777 " << end - start << endl;
    start = GetTimeStamp();

    std::string alignmentHash="";
    for (int i = 0; i < total; i++) {

        // aggregrate result
        #ifdef DEBUG
            cout << "< " << alignmentHash << endl;
            cout << ">("<< answers_hash[i].size() <<") " << answers_hash[i] << endl;
        #endif // DEBUG
        alignmentHash = sw::sha512::calculate(alignmentHash.append(answers_hash[i]));
        #ifdef DEBUG
            cout << alignmentHash << endl;
            std::cout << std::endl;
        #endif // DEBUG
    }

    end = GetTimeStamp();
    cout << "88888888 " << end - start << endl;
    start = GetTimeStamp();

    delete[] dp[0];
    delete[] dp;

    end = GetTimeStamp();
    cout << "99999999 " << end - start << endl;

	return alignmentHash;
}

// called for all tasks with rank!=root
// do stuff for each MPI task based on rank
void do_MPI_task(int rank) {
    uint64_t start = GetTimeStamp(), end;
    int n_threads = omp_get_max_threads();
    // cout << "rank[" << rank << "] has threads: " << n_threads << endl;
    omp_set_num_threads(n_threads);
    MPI_Status status;
    int size;
    MPI_Comm_size(comm, &size);

    int k_pxy_pgap[3];
    MPI_Bcast(k_pxy_pgap, 3, MPI_INT, root, comm);
    int k = k_pxy_pgap[0], 
        pxy = k_pxy_pgap[1],
        pgap = k_pxy_pgap[2];
    // cout << "rank[" << rank << "] received k pxy pgap " << endl;
    #ifdef DEBUG
        cout << "rank[" << rank << "][recv] " << "k: " << k << ", pxy: " << pxy << ", pgap: " << pgap << endl;
    #endif // DEBUG

    int total = k * (k-1) / 2;

    // number of dp matrix calculation per process
    int tasks_per_process = (int) floor((1.0*total) / size);
    int my_tasks_start = tasks_per_process * (rank-1), my_tasks_end = tasks_per_process * rank;

    // cout << "rank[" << rank << "] done task allocation " << n_threads << endl;
    int local_genes_len[k];
    MPI_Bcast(local_genes_len, k, MPI_INT, root, comm);

    // cout << "rank[" << rank << "] received string lengths " << n_threads << endl;
    
    int max_gene_len = *std::max_element(local_genes_len, local_genes_len + k) + 1;
    string local_genes[k];
    // #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < k; i++) {
        char buffer[max_gene_len];
        MPI_Bcast(buffer, local_genes_len[i], MPI_CHAR, root, comm);
        buffer[local_genes_len[i]] = '\0';
        local_genes[i] = string(buffer, local_genes_len[i]);
    }
    // cout << "rank[" << rank << "] received strings" << endl;

    

    int task_id = 0;
    // calculate calculation cells in each task and distribute evenly
    unsigned long long n_cells[total];
    unsigned long long total_cells = 0;
    for(int i=1;i<k;i++){
		for(int j=0;j<i;j++){
            n_cells[task_id] = ((long) local_genes_len[i]) * ((long) local_genes_len[j]);
            total_cells += n_cells[task_id];
            task_id++;
        }
    }

    unsigned long long cells_per_proccess = total_cells / size;
    vector<Triple> tasks[size]; // i, j, id of (i, j) in whole tasks
    task_id = 0;
    // int task_rank_mapping[total];
    unsigned long long rank_load[size];
    for (int i = 0; i < size; i++) {
        rank_load[i] = 0;
    }
    for(int i=1;i<k;i++){
		for(int j=0;j<i;j++){
            // make rank 0 as minimum task load as possible
            for (int r = size-1; r >= 0; r-- ) {
                // load task to rank
                if (rank_load[r] + n_cells[task_id] <= cells_per_proccess) {
                    rank_load[r] += n_cells[task_id];
                    tasks[r].push_back({ i, j, task_id });
                }
            }
            task_id++;
        }
    }

    int i_max_length = -1, j_max_length = -1;
    for (int z = 0; z < tasks[rank].size(); z++) {
        Triple t = tasks[rank].at(z);
        if (local_genes_len[t.x] > i_max_length) {
            i_max_length = local_genes_len[t.x];
        }
        if (local_genes_len[t.y] > j_max_length) {
            j_max_length = local_genes_len[t.y];
        }
    }
    

    // create dp matrix for all tasks
    int **dp = new2d(i_max_length + 1, j_max_length + 1);
    // intialising the table
    #pragma omp parallel 
    {
        #pragma omp for nowait
        for (int i = 0; i <= i_max_length; i++) {
            dp[i][0] = i * pgap;
        }
        #pragma omp for
        for (int i = 1; i <= j_max_length; i++) {
            dp[0][i] = i * pgap;
        }
    }

    start = GetTimeStamp();
    // do sequence alignment calculation
    int n_tasks = tasks[rank].size();
    int task_ids[n_tasks];
    int task_penalties[n_tasks];
    string task_problemhashs[n_tasks];
    for (int z = 0; z < n_tasks; z++) {
        Triple xyz = tasks[rank].at(z);
        int i = xyz.x, j = xyz.y;
        task_ids[z] = xyz.z;
        int l = local_genes_len[i] + local_genes_len[j];
        int xans[l+1], yans[l+1];
        task_penalties[z] = getMinimumPenalty2(local_genes[i], local_genes[j], pxy, pgap, xans, yans, local_genes_len[i], local_genes_len[j], dp);
        // Since we have assumed the answer to be n+m long,
        // we need to remove the extra gaps in the starting
        // id represents the index from which the arrays
        // xans, yans are useful
        int id = 1;
        int a;
        for (a = l; a >= 1; a--) {
            if ((char)yans[a] == '_' && (char)xans[a] == '_') {
                id = a + 1;
                break;
            }
        }
        std::string align1="";
        std::string align2="";
        for (a = id; a <= l; a++) {
            align1.append(1,(char)xans[a]);
        }
        for (a = id; a <= l; a++) {
            align2.append(1,(char)yans[a]);
        }
        std::string align1hash = sw::sha512::calculate(align1);
        std::string align2hash = sw::sha512::calculate(align2);
        std::string problemhash = sw::sha512::calculate(align1hash.append(align2hash));
        // store problemhash sent to root
        task_problemhashs[z] = problemhash;
        #ifdef DEBUG
            cout << "rank[" << rank << "][calc] " << "task id: " << task_ids[z] << ", penalty: " << task_penalties[z] << ", hash("<< task_problemhashs[z].size() <<"): " << task_problemhashs[z] << endl;
        #endif // DEBUG
    }    

    end = GetTimeStamp();
    cout << "66666666 rank[" << rank << "]" << end - start << endl;
    // start = GetTimeStamp();

    // cout << "rank[" << rank << "] finish calculations " << endl;

    // MPI_Send(task_ids, n_tasks, MPI_INT, root, collect_results_tag, comm);
    MPI_Send(&n_tasks, 1, MPI_INT, root, collect_results_tag, comm);
    MPI_Send(task_penalties, n_tasks, MPI_INT, root, collect_results_tag2, comm);
    // sent jobs result to root
    for (int i = 0; i < n_tasks; i++) {
        MPI_Send(task_problemhashs[i].c_str(), 128, MPI_CHAR, root, collect_results_tag3, comm);
    }

    delete[] dp[0];
    delete[] dp;

    // cout << "rank[" << rank << "] finish send results to root " << endl;
    #ifdef DEBUG
        cout << "rank[" << rank << "][finish] " << endl;
    #endif // DEBUG
}

int getMinimumPenalty2(std::string x, std::string y, int pxy, int pgap, int *xans, int *yans, int m, int n, int **dp) {
	int i, j; // intialising variables
    int n_threads = omp_get_num_threads();

    // int m = x.length(); // length of gene1
    // int n = y.length(); // length of gene2
    int row = m + 1, col = n + 1;

    // table for storing optimal substructure answers
    // int **dp = new2d(row, col);
//	size_t size = m + 1;
//	size *= n + 1;
//	memset (dp[0], 0, size);

    #ifdef DEBUG
		cout.fill(' ');
        for (i = 0; i < row; i++) {
            for (j = 0; j < col; j++) {
                // Prints ' ' if j != n-1 else prints '\n'           
                cout << setw(3) << dp[i][j] << " "; 
			}
			cout << "\n";
		}
        cout << ">>>> \n";
    #endif

    // calcuting the minimum penalty
    
    // Tile parallel
    int n_parallel = n_threads + (int (floor((1.0 * n_threads) / 3)));
    // calculate tile size
    int tile_width  = (int) ceil((1.0*m) / n_parallel), 
        tile_length = (int) ceil((1.0*n) / n_parallel);
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

    #ifdef DEBUG
		cout.fill(' ');
        for (i = 0; i < row; i++) {
            for (j = 0; j < col; j++) {         
                cout << setw(3) << dp[i][j] << " "; 
			}
			cout << "\n";
		}
        cout << ">>>> \n";
    #endif

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
    while (xpos > 0) {
        if (i > 0) xans[xpos--] = (int) x[--i];
        else xans[xpos--] = (int) '_';
    }
    while (ypos > 0) {
        if (j > 0) yans[ypos--] = (int) y[--j];
        else yans[ypos--] = (int) '_';
    }

    int ret = dp[m][n];

    // delete[] dp[0];
    // delete[] dp;

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
