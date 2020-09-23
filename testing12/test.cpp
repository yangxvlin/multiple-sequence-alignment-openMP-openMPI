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
#include <queue> 
#include <map>
#include <algorithm>    // std::sort

// uncomment to enable debug
// #define debug 0

int getMinimumPenalty2(std::string x, std::string y, int pxy, int pgap, int *xans, int *yans, int m, int n);

// const int n_threads = 16;
const int sha512_strlen = 128 + 1; // +1 for '\0'

const int ask_for_genes_tag = 1;
const int send_genes_tag = 2;
const int collect_results_tag = 3;
const int collect_results_tag2 = 4;
const int collect_results_tag3 = 5;
const int new_task_flag = 6;

struct Triple { 
   int x, y, z; 
}; 

struct Task {
    int i, j, task_id;
    float task_cost; 
};
struct task_cost_cmp {
    bool operator()(const Task & a, const Task & b) {
        // largest comes first
        return a.task_cost > b.task_cost;
    }
};

struct Quatic {
    int x, y, z, r; 
};

struct Packet {
    int task_id;
    int task_penalty;
    char task_hash[sha512_strlen];
};

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

// called by the root MPI task only
// this procedure should distribute work to other MPI tasks
// and put together results, etc.
std::string getMinimumPenalties(std::string *genes, 
                                       int k, 
                                       int pxy, 
                                       int pgap,
	                                   int *penalties) {
    int n_threads = omp_get_max_threads();
    // cout << "rank[" << 0 << "] has threads: " << n_threads << endl;
    omp_set_num_threads(n_threads);

    map< pair<int, int> , float> cost_map;
    cost_map[{ 30000,90000 }] = 2685443;
    cost_map[{ 50000,90000 }] = 3368568;
    cost_map[{ 40000,90000 }] = 3820399;
    cost_map[{ 45000,90000 }] = 4346901;
    cost_map[{ 35000,90000 }] = 4562313;
    cost_map[{ 55000,90000 }] = 4661260;
    cost_map[{ 80000,90000 }] = 6844260;
    cost_map[{ 85000,90000 }] = 6981953;
    cost_map[{ 45000,80000 }] = 3689918;
    cost_map[{ 80000,85000 }] = 5879212;
    cost_map[{ 45000,85000 }] = 5238173;
    cost_map[{ 70000,90000 }] = 7109908;
    cost_map[{ 40000,80000 }] = 3401441;
    cost_map[{ 65000,90000 }] = 6808658;
    cost_map[{ 75000,90000 }] = 6519042;
    cost_map[{ 60000,90000 }] = 4448726;
    cost_map[{ 65000,85000 }] = 4963178;
    cost_map[{ 50000,85000 }] = 3186950;
    cost_map[{ 35000,80000 }] = 2946756;
    cost_map[{ 35000,85000 }] = 3278903;
    cost_map[{ 40000,85000 }] = 2614118;
    cost_map[{ 50000,80000 }] = 5846400;
    cost_map[{ 55000,85000 }] = 3484993;
    cost_map[{ 30000,85000 }] = 1898632;
    cost_map[{ 75000,80000 }] = 5859720;
    cost_map[{ 75000,85000 }] = 5759294;
    cost_map[{ 70000,85000 }] = 6523608;
    cost_map[{ 60000,85000 }] = 5030286;
    cost_map[{ 35000,70000 }] = 1929892;
    cost_map[{ 35000,75000 }] = 2018626;
    cost_map[{ 30000,80000 }] = 1753262;
    cost_map[{ 45000,70000 }] = 2455557;
    cost_map[{ 30000,75000 }] = 1781182;
    cost_map[{ 70000,80000 }] = 6568076;
    cost_map[{ 60000,80000 }] = 3963122;
    cost_map[{ 45000,75000 }] = 3679246;
    cost_map[{ 55000,80000 }] = 6485492;
    cost_map[{ 45000,65000 }] = 2335015;
    cost_map[{ 65000,75000 }] = 4239867;
    cost_map[{ 65000,80000 }] = 5710165;
    cost_map[{ 40000,75000 }] = 2232920;
    cost_map[{ 30000,70000 }] = 2422724;
    cost_map[{ 60000,70000 }] = 3866669;
    cost_map[{ 55000,60000 }] = 2493375;
    cost_map[{ 55000,70000 }] = 3112889;
    cost_map[{ 30000,65000 }] = 1572207;
    cost_map[{ 40000,70000 }] = 4042016;
    cost_map[{ 30000,50000 }] = 1193198;
    cost_map[{ 60000,75000 }] = 5065149;
    cost_map[{ 50000,75000 }] = 3796500;
    cost_map[{ 65000,70000 }] = 3830619;
    cost_map[{ 50000,70000 }] = 4017638;
    cost_map[{ 45000,60000 }] = 1803092;
    cost_map[{ 35000,55000 }] = 1619453;
    cost_map[{ 40000,45000 }] = 1463394;
    cost_map[{ 30000,60000 }] = 1373043;
    cost_map[{ 35000,60000 }] = 1986794;
    cost_map[{ 30000,55000 }] = 1358983;
    cost_map[{ 50000,55000 }] = 2037774;
    cost_map[{ 35000,40000 }] = 1200278;
    cost_map[{ 35000,65000 }] = 2752363;
    cost_map[{ 40000,65000 }] = 1749777;
    cost_map[{ 60000,65000 }] = 4368559;
    cost_map[{ 40000,55000 }] = 1577821;
    cost_map[{ 55000,65000 }] = 3613808;
    cost_map[{ 30000,40000 }] = 1004363;
    cost_map[{ 50000,60000 }] = 3488488;
    cost_map[{ 70000,75000 }] = 7632558;
    cost_map[{ 50000,65000 }] = 3547563;
    cost_map[{ 45000,50000 }] = 2376754;
    cost_map[{ 30000,35000 }] = 949612 ;
    cost_map[{ 35000,50000 }] = 1813099;
    cost_map[{ 45000,55000 }] = 2995864;
    cost_map[{ 30000,45000 }] = 1276153;
    cost_map[{ 35000,45000 }] = 1731105;
    cost_map[{ 55000,75000 }] = 5984097;
    cost_map[{ 40000,60000 }] = 3623866;
    cost_map[{ 40000,50000 }] = 2949727;

    // number of processes
    int size;
    MPI_Comm_size(comm, &size);
    // number of workers
    int n_workers = size - 1;

    // broadcast k, pxy, pgap to wrokers
    int k_pxy_pgap[3] = {k, pxy, pgap};
    MPI_Bcast(k_pxy_pgap, 3, MPI_INT, root, comm);

    // total tasks
    int total = k * (k-1) / 2;
    // calculates string length
    int genes_length[k];
    int genes_approx_length[k];
    for (int i = 0; i < k; i++) {
        genes_length[i] = genes[i].length();
        genes_approx_length[i] = genes_length[i] / 5000 * 5000;
    }
    // broadcast strings length to wrokers
    MPI_Bcast(genes_length, k, MPI_INT, root, comm);

    // broadcast strings to wrokers
    int max_gene_len = *std::max_element(genes_length, genes_length + k) + 1;
    char str_buffer[max_gene_len];
    for (int i = 0; i < k; i++) {
        memcpy(str_buffer, genes[i].c_str(), genes_length[i]);
        MPI_Bcast(str_buffer, genes_length[i], MPI_CHAR, root, comm);
    }

    // load remaining tasks
    priority_queue<Task, vector<Task>, task_cost_cmp> remaining_tasks;
    int n_remaining_tasks = total - n_workers;
    int task_id = 0;
    int i_approx, j_approx;
    for(int i=1;i<k;i++){
		for(int j=0;j<i;j++){
            // worker's default tasks ignored
            if (task_id >= n_workers) {
                Task t;
                t.i = i;
                t.j = j;
                t.task_id = task_id;
                i_approx = max(genes_approx_length[i], 30000);
                j_approx = max(genes_approx_length[j], 35000);
                t.task_cost = cost_map[{ ,  }];
                remaining_tasks.push(t);
            }
            task_id++;
        }
    }

    // master's dynamic task control
    MPI_Status status;
    int task_penalty, task_source;
    task_id = 0;
    string answers_hash[total];
    int i_j_task_id[3];
    for (int t = 0; t < n_remaining_tasks; t++) {
        // recv task id
        MPI_Recv(&task_id, 1, MPI_INT, MPI_ANY_SOURCE, collect_results_tag, comm, &status);
        task_source = status.MPI_SOURCE;
        // recv task penalty
        MPI_Recv(&task_penalty, 1, MPI_INT, task_source, collect_results_tag2, comm, &status);
        penalties[task_id] = task_penalty;
        // recv task answer
        char answer_buffer[sha512_strlen];
        MPI_Recv(answer_buffer, 128, MPI_CHAR, task_source, collect_results_tag3, comm, &status);
        answers_hash[task_id] = string(answer_buffer, 128);
        cout << "rank[0] from rank[" << task_source << "]: task id: " << task_id << ", penalty: " << task_penalty << ", hash: " << answers_hash[task_id] << endl;

        // has tasak for worker
        if (!remaining_tasks.empty()) {
            Task t = remaining_tasks.top();
            i_j_task_id[0] = t.i;
            i_j_task_id[1] = t.j;
            i_j_task_id[2] = t.task_id;
            remaining_tasks.pop();
            MPI_Send(i_j_task_id, 3, MPI_INT, task_source, new_task_flag, comm);
            cout << "rank[0] more task for rank[" << task_source << "]: task id:" << t.task_id << " (" << t.i << ", " << t.j << ") " << "cost: " << t.task_cost << endl;
        // no more task for worker
        } else {
            i_j_task_id[0] = -1;
            i_j_task_id[1] = -1;
            i_j_task_id[2] = -1;
            MPI_Send(i_j_task_id, 3, MPI_INT, task_source, new_task_flag, comm);
            cout << "rank[0] no more task for rank[" << task_source << endl;
        }
    }

    // cout << "77777777" << endl;

    std::string alignmentHash="";
    for (int i = 0; i < total; i++) {

        // aggregrate answers
        alignmentHash = sw::sha512::calculate(alignmentHash.append(answers_hash[i]));
    }

	return alignmentHash;
}

// called for all tasks with rank!=root
// do stuff for each MPI task based on rank
void do_MPI_task(int rank) {
    int n_threads = omp_get_max_threads();
    // cout << "rank[" << rank << "] has threads: " << n_threads << endl;
    omp_set_num_threads(n_threads);
    MPI_Status status;
    
    // number of processes
    int size;
    MPI_Comm_size(comm, &size);
    // number of workers
    int n_workers = size - 1;

    // broadcast strings length from master
    int k_pxy_pgap[3];
    MPI_Bcast(k_pxy_pgap, 3, MPI_INT, root, comm);
    int k = k_pxy_pgap[0], 
        pxy = k_pxy_pgap[1],
        pgap = k_pxy_pgap[2];

    // total tasks
    int total = k * (k-1) / 2;

    // broadcast strings length from master
    int local_genes_len[k];
    MPI_Bcast(local_genes_len, k, MPI_INT, root, comm);
    
    // broadcast strings from master
    int max_gene_len = *std::max_element(local_genes_len, local_genes_len + k) + 1;
    string local_genes[k];
    for (int i = 0; i < k; i++) {
        char buffer[max_gene_len];
        MPI_Bcast(buffer, local_genes_len[i], MPI_CHAR, root, comm);
        buffer[local_genes_len[i]] = '\0';
        local_genes[i] = string(buffer, local_genes_len[i]);
    }

    // initial default task
    Triple task;
    int task_id = 1;
    bool flag = true, has_more_work = false;
    int n_tasks_done = 0;
    for(int i=1;i<k && flag;i++){
		for(int j=0;j<i;j++){
            if (task_id == rank) {
                task.x = i;
                task.y = j;
                task.z = task_id;
                flag = false;
                has_more_work = true;
                break;
            }
            task_id++;
        }
    }

    // worker works
    int i, j, l, task_penalty;
    int i_j_task_id[3];
    while (has_more_work) {
        // do initial default task
        if (n_tasks_done == 0) {
            i = task.x, j = task.y, task_id = task.z;
        // more tasks from master
        } else {
            MPI_Recv(i_j_task_id, 3, MPI_INT, root, new_task_flag, comm, &status);
            i = i_j_task_id[0];
            j = i_j_task_id[1];
            task_id = i_j_task_id[2];
        }

        // worker terminate
        if (task_id == -1) {
            has_more_work = false;
            break;
        }
        
        // do task: sequence alignment calculation
        int l = local_genes_len[i] + local_genes_len[j];
        int xans[l+1], yans[l+1];
        task_penalty = getMinimumPenalty2(local_genes[i], local_genes[j], pxy, pgap, xans, yans, local_genes_len[i], local_genes_len[j]);
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
        
        MPI_Send(&task_id, 1, MPI_INT, root, collect_results_tag, comm);
        MPI_Send(&task_penalty, 1, MPI_INT, root, collect_results_tag2, comm);
        MPI_Send(problemhash.c_str(), 128, MPI_CHAR, root, collect_results_tag3, comm);
        n_tasks_done++;
    }
}

int getMinimumPenalty2(std::string x, std::string y, int pxy, int pgap, int *xans, int *yans, int m, int n) {
	int i, j; // intialising variables
    int n_threads = omp_get_num_threads();

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
		if (i > 0) xans[xpos--] = (int)x[--i];
		else xans[xpos--] = (int)'_';
	}
	while (ypos > 0) {
		if (j > 0) yans[ypos--] = (int)y[--j];
		else yans[ypos--] = (int)'_';
	}

    // int x_diff = xpos - i, y_diff = ypos - j;
    // #pragma omp parallel for
    // for (int ii = i; ii>0; ii--){
    //     xans[ii + x_diff] = (int)x[ii - 1];
    // }
    // #pragma omp parallel for
    // for (int x_dash = x_diff; x_dash>0; x_dash--){
    //     xans[x_dash] = (int)'_';
    // }

    // #pragma omp parallel for
    // for (int jj = j; jj>0; jj--){
    //     yans[jj + y_diff] = (int)y[jj - 1];
    // }

    // #pragma omp parallel for
    // for (int y_dash = y_diff; y_dash>0; y_dash--){
    //     yans[y_dash] = (int)'_';
    // }

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
