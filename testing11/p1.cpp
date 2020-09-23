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
#include "omp.h"
#include "math.h"
#include <vector>
#include <tuple>
#include <algorithm>
#include <queue> 

int n_jobs_distribute_tag = 0;
int my_job_distribute_tag = 1;
int result_collect_tag = 2;

// define the type for a job
typedef struct {
  int i, j, id;
} JOB_t;

// define the type for a result
typedef struct{
    int penalty, id;
    char problemhash[129];
} RESULT_t;

inline int min3(int a, int b, int c) {
	if (a <= b) {
        if (a <= c) {
            return a;
        }
        return c;
    } else if (b <= c) {
        return b;
    }
    return c;
}

// equivalent of  int *dp[width] = new int[height][width]
// but works for width not known at compile time.
// (Delete structure by  delete[] dp[0]; delete[] dp;)
inline int **new2d (int width, int height)
{
	int **dp = new int *[width];
	size_t size = width;
	size *= height;
	int *dp0 = new int [size];
	if (!dp || !dp0)
	{
	    std::cerr << "getMinimumPenalty: new failed" << std::endl;
	    exit(1);
	}
	dp[0] = dp0;
	for (int i = 1; i < width; i++)
	    dp[i] = dp[i-1] + height;

	return dp;
}

inline bool compareByJobId(const RESULT_t &a, const RESULT_t &b){
    return a.id < b.id;
}

// called by the root MPI task **only**
// this procedure should distribute work to other MPI tasks
// and put together results, etc.
std::string getMinimumPenalties(std::string *genes, int k, int pxy, int pgap,
	int *penalties)
{

    int size;
    MPI_Comm_size(comm, &size);

    // define the job type for MPI 
    MPI_Datatype MPI_JOB;
    MPI_Type_contiguous(3, MPI_INT, &MPI_JOB);
    MPI_Type_commit(&MPI_JOB);

    // Broadcast the 3 configuration number
    int config[3] = {k, pxy, pgap};// k, pxy, pgap
    MPI_Bcast(config, 3, MPI_INT, root, comm);


    // Broadcast the sequence length list
    int seq_length[k];

    for (int i=0; i<k; ++i){
        seq_length[i] = genes[i].length();
    }
    MPI_Bcast(seq_length, k, MPI_INT, root, comm);

    
    // Broadcast the sequences 
    for (int i = 0; i < k; i++) {
        char buffer[seq_length[i]];
        memcpy(buffer, genes[i].c_str(), seq_length[i]);
        MPI_Bcast(buffer, seq_length[i], MPI_CHAR, root, comm);
    }

    // create job id (#cell, <i, j, job-id>)



    vector<pair<float, JOB_t>> jobs;
    int job_id = 0;
	for(int i=1;i<k;i++){
		for(int j=0;j<i;j++){
            float size = (((float)seq_length[i])/1000.0)*8.1 + (((float)seq_length[j])/1000.0)*4.1 - 334.3;
            jobs.push_back({size, {i, j, job_id++}});
            
            // jobs.push_back({(seq_length[i/10000])*(seq_length[j]/10000), {i, j, job_id++}});
        }
    }

    // sort the jobs vector on the #cell
    std::sort(jobs.begin(), jobs.end(), [](auto &left, auto &right) {
        return left.first > right.first;
    });

    // // for(int i=0; i < jobs.size(); i++){
    // //     // jobs[i].doSomething();
    // //     cout << "problem size in cell is " << jobs[i].first << endl;
    // //     cout << "i=" << get<0>(jobs[i].second) << ", j=" << get<1>(jobs[i].second) << "job-id=" << get<2>(jobs[i].second) << endl;
    // // }

    // distribute the jobs evenly across all workers with respect to the # cell
    typedef pair<float, vector<JOB_t>> P;

    struct CompareByFirst {
        constexpr bool operator()(P const & a,
                              P const & b) const noexcept
        { return a.first  >  b.first; }
    };

    priority_queue<P, vector<P>, CompareByFirst> queue;

    for (int i=0; i<size; i++){
        queue.push({0, {}});
    }

    for (int i=0; i < jobs.size(); i++){
        // cout << queue.top().first << endl;
        pair<float, vector<JOB_t>> partition = queue.top();
        queue.pop();

        partition.second.push_back(jobs[i].second);    

        partition.first += jobs[i].first;

        // break;
        queue.push(partition);
        // cout << queue.top().first << endl;

        // get the min worker
        
        // add a new to the worker

        // put the worker back, update the priority queue

    }

    // check partitions eveness
    // for (int i=0; i<size; i++){
    //     pair<int, vector<JOB_t>> partition = queue.top();
    //     queue.pop();

    //     cout << partition.second.size() << endl;
    // }


    // take the last partition for the root worker
    vector<JOB_t> my_jobs = queue.top().second;
    cout << "rank-" << root << ": " << queue.top().first << endl;
    queue.pop();

    
    int n_jobs_list[size];
    // distribute job partition to each worker
    for (int i=1; i<size; i++){
        pair<float, vector<JOB_t>> partition = queue.top();
        queue.pop();

        cout << "rank-" << i << ": " << partition.first << endl; 
        // send to worker i
        int n_jobs = partition.second.size(); 
        MPI_Send(&n_jobs, 1, MPI_INT, i, n_jobs_distribute_tag, comm);
        n_jobs_list[i] = n_jobs;
        MPI_Send(&partition.second[0], n_jobs, MPI_JOB, i, my_job_distribute_tag, comm);
    }

    // receive result vectors from wokers
    // define the job type for MPI 
    MPI_Datatype MPI_SHA512;
    MPI_Type_contiguous(128, MPI_CHAR, &MPI_SHA512);
    MPI_Type_commit(&MPI_SHA512);



    MPI_Datatype MPI_RESULT;
    int blen[3];
    MPI_Aint array_of_displacements[3];
    MPI_Datatype oldtypes[3];
    blen[0] = 1; array_of_displacements[0] = offsetof(RESULT_t, penalty); oldtypes[0] = MPI_INT;
    blen[1] = 1; array_of_displacements[1] = offsetof(RESULT_t, id); oldtypes[1] = MPI_INT;
    blen[2] = 129; array_of_displacements[2] = offsetof(RESULT_t, problemhash); oldtypes[2] = MPI_CHAR;
    MPI_Type_create_struct( 3, blen, array_of_displacements, oldtypes, &MPI_RESULT );
    MPI_Type_commit(&MPI_RESULT);

    MPI_Status status;

    vector<RESULT_t> results = {};

    uint64_t matrix_start = GetTimeStamp();
    // do the jobs
    int n_jobs = my_jobs.size();
    for (int z=0; z<n_jobs; ++z){   
        uint64_t start_ = GetTimeStamp();

        int i=my_jobs[z].i;
        int j=my_jobs[z].j;
        int job_id=my_jobs[z].id;

        std::string gene1 = genes[i];
        std::string gene2 = genes[j];
        int m = gene1.length(); // length of gene1
        int n = gene2.length(); // length of gene2
        int l = m+n;
        int xans[l+1], yans[l+1];
        int penalty = getMinimumPenalty(gene1,gene2,pxy,pgap,xans,yans);

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
        std::string align1="";
        std::string align2="";
        for (a = id; a <= l; a++)
        {
            align1.append(1,(char)xans[a]);
        }
        for (a = id; a <= l; a++)
        {
            align2.append(1,(char)yans[a]);
        }

        // alignmentHash = hash(alignmentHash ++ hash(hash(align1)++hash(align2)))
        std::string align1hash = sw::sha512::calculate(align1);
        std::string align2hash = sw::sha512::calculate(align2);
        std::string problemhash = sw::sha512::calculate(align1hash.append(align2hash));

        RESULT_t result;
        result.penalty = penalty;
        result.id = job_id;
        strcpy(result.problemhash, problemhash.c_str());
        results.push_back(result);
        
        uint64_t end_ = GetTimeStamp();
        // cout << "rank==" << root << ", task-id = "<< job_id << ", i=" << i << "(" << seq_length[i] << ")" << ", j=" << j << "(" << seq_length[j] << ")" <<  "  " << end_ - start_  << endl;
        // cout << seq_length[i] << "," << seq_length[j] << "," << end_ - start_   << endl;
    
    }


    uint64_t end = GetTimeStamp();
    cout << "rank-" << root << " computes " << end - matrix_start << "us for matrix "<< endl;

    for (int i=1; i<size; ++i){
        RESULT_t temp[n_jobs_list[i]];
        MPI_Recv(&temp, n_jobs_list[i], MPI_RESULT, i, result_collect_tag, comm, &status);

        for (int j=0; j<n_jobs_list[i]; ++j){
            results.push_back(temp[j]);
        }
    } 

    // sort the result on the job-id
    std::sort(results.begin(), results.end(), compareByJobId);

    std::string alignmentHash="";
    for (int i=0; i<results.size();++i){
        alignmentHash=sw::sha512::calculate(alignmentHash.append(results[i].problemhash));
        penalties[i]=results[i].penalty;
        // cout << results[i].id << endl;
    }

    
    

    // for (int i=1; i<size-1; ++i){
    //     cout << "receiving " << i << endl;
    //     MPI_Recv(&results[i], n_jobs_list[i], MPI_RESULT, i, result_collect_tag, comm, &status);
    // } 
    

    // for (int i=0; i<2; ++i){
    //     cout << "rank-" << root << ": penalty=" << test[i].penalty << ", id=" << test[i].id << ", problemhash=" << test[i].problemhash << endl;
    
    // }
    
	return alignmentHash;
}

// called for all tasks with rank!=root
// do stuff for each MPI task based on rank
void do_MPI_task(int rank)
{


    // define the job type for MPI 
    MPI_Datatype MPI_JOB;
    MPI_Type_contiguous(3, MPI_INT, &MPI_JOB);
    MPI_Type_commit(&MPI_JOB);

    MPI_Status status;

    // receive the configuration
    int config[3]; // k, pxy, pgap
    MPI_Bcast(config, 3, MPI_INT, root, comm);

    int misMatchPenalty = config[1];
    int gapPenalty = config[2];
    int k = config[0];


    // receive the sequneces length list
    int seq_length[k];
    MPI_Bcast(seq_length, k, MPI_INT, root, comm);

    // receive the gene sequences
    string genes[k];
    for (int i = 0; i < k; i++) {
        char buffer[seq_length[i]+1];
        MPI_Bcast(buffer, seq_length[i], MPI_CHAR, root, comm);
        buffer[seq_length[i]] = '\0';
        genes[i] = string(buffer, seq_length[i]);
        // cout << "rank " << rank << "  " << genes[i] << endl;
    }

    // receive my jobs

    vector<JOB_t> my_jobs;
    int n_jobs;
    
    MPI_Recv(&n_jobs, 1, MPI_INT, root, n_jobs_distribute_tag, comm, &status); 
    my_jobs.resize(n_jobs);

    
    // MPI_Status status;
    MPI_Recv(&my_jobs[0], n_jobs, MPI_JOB, root, my_job_distribute_tag, comm, &status); 
    // for (int i=0; i<n_jobs; ++i){
    //     cout << "rank-" << rank << ": i=" << my_jobs[i].i << ", j=" << my_jobs[i].j << ", job-id=" << my_jobs[i].id << endl;
    // }

    
    uint64_t start = GetTimeStamp();
    // do the jobs
    vector<RESULT_t> my_result;
    for (int z=0; z<n_jobs; ++z){
        uint64_t start_ = GetTimeStamp();

        int i=my_jobs[z].i;
        int j=my_jobs[z].j;
        int job_id=my_jobs[z].id;

        std::string gene1 = genes[i];
        std::string gene2 = genes[j];
        int m = gene1.length(); // length of gene1
        int n = gene2.length(); // length of gene2
        int l = m+n;
        int xans[l+1], yans[l+1];
        int penalty = getMinimumPenalty(gene1,gene2,misMatchPenalty,gapPenalty,xans,yans);

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
        std::string align1="";
        std::string align2="";
        for (a = id; a <= l; a++)
        {
            align1.append(1,(char)xans[a]);
        }
        for (a = id; a <= l; a++)
        {
            align2.append(1,(char)yans[a]);
        }

        // alignmentHash = hash(alignmentHash ++ hash(hash(align1)++hash(align2)))
        std::string align1hash = sw::sha512::calculate(align1);
        std::string align2hash = sw::sha512::calculate(align2);
        std::string problemhash = sw::sha512::calculate(align1hash.append(align2hash));

        RESULT_t result;
        result.penalty = penalty;
        result.id = job_id;
        strcpy(result.problemhash, problemhash.c_str());
        my_result.push_back(result);


        uint64_t end_ = GetTimeStamp();
        // cout << seq_length[i] << "," << seq_length[j] << "," << end_ - start_   << endl;
        // cout << "rank==" << rank << ", task-id = "<< job_id << ", i=" << i << "(" << seq_length[i] << ")" << ", j=" << j << "(" << seq_length[j] << ")" <<  "  " << end_ - start_  << endl;
    }

    uint64_t end = GetTimeStamp();
    cout << "rank-" << rank << " computes " << end - start << "us for matrix "<< endl;

    // for (int i=0; i<n_jobs; ++i){
    //     cout << "rank-" << rank << ": penalty=" << my_result[i].penalty << ", id=" << my_result[i].id << ", problemhash=" << my_result[i].problemhash << endl;
    // }


    // Send the result vector back to the root process
    // define the job type for MPI 
    MPI_Datatype MPI_SHA512;
    MPI_Type_contiguous(129, MPI_CHAR, &MPI_SHA512);
    MPI_Type_commit(&MPI_SHA512);



    MPI_Datatype MPI_RESULT;
    int blen[3];
    MPI_Aint array_of_displacements[3];
    MPI_Datatype oldtypes[3];
    blen[0] = 1; array_of_displacements[0] = offsetof(RESULT_t, penalty); oldtypes[0] = MPI_INT;
    blen[1] = 1; array_of_displacements[1] = offsetof(RESULT_t, id); oldtypes[1] = MPI_INT;
    blen[2] = 129; array_of_displacements[2] = offsetof(RESULT_t, problemhash); oldtypes[2] = MPI_CHAR;
    MPI_Type_create_struct( 3, blen, array_of_displacements, oldtypes, &MPI_RESULT );
    MPI_Type_commit(&MPI_RESULT);
 

    MPI_Send(&my_result[0], n_jobs, MPI_RESULT, root, result_collect_tag, comm);
    
    // std::cin >> misMatchPenalty;
    // std::cin >> gapPenalty;
    // std::cin >> k;	
    // std::string genes[k];
    // for(int i=0;i<k;i++) std::cin >> genes[i];

    // cout << "misMatchPenalty: " << misMatchPenalty << ", gapPenalty: " << gapPenalty << ", k: " << k << endl;
    // for (int i=0; i<k; ++i){
    //     cout << " " << seq_length[i] << " ";
    // }
    // cout << endl;
    // cout << "string count: " << genes.size() << endl;
}

// function to find out the minimum penalty
// return the minimum penalty and put the aligned sequences in xans and yans
inline int getMinimumPenalty(std::string x, std::string y, int pxy, int pgap, int *xans, int *yans)
{
	
	int i, j; // intialising variables

    int m = x.length(); // length of gene1
    int n = y.length(); // length of gene2

    // table for storing optimal substructure answers
    omp_set_num_threads(omp_get_max_threads());
    int **dp = new2d(m + 1, n + 1);


    // remove unnecessary memset
//    size_t size = m + 1;
//    size *= n + 1;
//    memset (dp[0], 0, size);

    // intialising the table
#pragma omp parallel for
    for (i = 0; i <= m; ++i) {
        dp[i][0] = i * pgap;
    }
#pragma omp parallel for
    for (i = 0; i <= n; ++i) {
        dp[0][i] = i * pgap;
    }

    int n_threads = 16;
    omp_set_num_threads(n_threads);
    // calculating the minimum penalty with the tiling technique in an anti-diagonal version
    int tile_row_size = (int) ceil((1.0 * m) / n_threads); // Number of dp elements in row of each tile
    int tile_col_size = (int) ceil((1.0 * n) / n_threads); // Number of dp elements in column of each tile

//    int tile_row_size = 256; // Number of dp elements in row of each tile
//    int tile_col_size = 256; // Number of dp elements in column of each tile
    int tile_m = (int) ceil((1.0 * m) / tile_row_size); // Number of tiles in row of the dp matrix
    int tile_n = (int) ceil((1.0 * n) / tile_col_size); // Number of tile in column of the dp matrix

    int total_diagonal = tile_m + tile_n - 1;
    int row_min, row_max, diagonal_index, k;
//    cout << "tile_row_size: " << tile_row_size << ", tile_col_size: " << tile_col_size << endl;
//    cout << "tile_m: " << tile_m << ", tile_n: " << tile_n << endl;
//    cout << "total_diagonal: " << total_diagonal << endl;
    for (diagonal_index = 1; diagonal_index <= total_diagonal; ++diagonal_index) {
        row_min = max(1, diagonal_index - tile_n + 1);
        row_max = min(diagonal_index, tile_m);
#pragma omp parallel for
        for (k = row_min; k <= row_max; ++k) {
            int tile_row_start = 1 + (k - 1) * tile_row_size; // index inclusive
            int tile_row_end = min(tile_row_start + tile_row_size, m + 1); // index exclusive
            int tile_col_start = 1 + (diagonal_index - k) * tile_col_size; // index inclusive
            int tile_col_end = min(tile_col_start + tile_col_size, n + 1); // index exclusive

//            cout << "(" << tile_row_start<< "," << tile_col_start << ")" << " | ";
//            cout << "-> (" << tile_row_end << "," << tile_col_end << ")" << '|';
            for (int ii = tile_row_start; ii < tile_row_end; ++ii) {
                for (int jj = tile_col_start; jj < tile_col_end; ++jj) {
                    if (x[ii - 1] == y[jj - 1]) {
                        dp[ii][jj] = dp[ii - 1][jj - 1];
                    } else {
                        dp[ii][jj] = min3(dp[ii - 1][jj - 1] + pxy,
                                          dp[ii - 1][jj] + pgap,
                                          dp[ii][jj - 1] + pgap);
                    }
                }
            }
        }
//        cout << "done" << endl;
    }


    // Reconstructing the solution
    int l = n + m; // maximum possible length

    i = m;
    j = n;

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
        else
        {
            xans[xpos--] = (int)'_';
            yans[ypos--] = (int)y[j - 1];
            j--;
        }
    }


    omp_set_num_threads(omp_get_max_threads());
    int x_diff = xpos - i, y_diff = ypos-j;
#pragma omp parallel for
    for (int ii = i; ii>0; --ii){
        xans[ii+x_diff] = (int)x[ii-1];
    }
#pragma omp parallel for
    for (int x_pos2=xpos-i; x_pos2>0; --x_pos2){
        xans[x_pos2] = (int)'_';
    }

#pragma omp parallel for
    for (int jj = j; jj>0; --jj){
        yans[jj+y_diff] = (int)y[jj-1];
        if (jj==0){
        }
    }

#pragma omp parallel for
    for (int y_pos2=ypos-j; y_pos2>0; --y_pos2){
        yans[y_pos2] = (int)'_';
    }

    int ret = dp[m][n];

    delete[] dp[0];
    delete[] dp;

    return ret;
}
