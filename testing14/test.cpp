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
uint64_t GetTimeStamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * (uint64_t)1000000 + tv.tv_usec;
}

const MPI_Comm comm = MPI_COMM_WORLD;
const int root = 0;

// Driver code
int main(int argc, char **argv)
{
    int rank;

    int prov;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &prov);
    // cout << "threads: " << prov << endl;

    MPI_Comm_rank(comm, &rank);
    if (rank == root)
    {
        int misMatchPenalty;
        int gapPenalty;
        int k;
        std::cin >> misMatchPenalty;
        std::cin >> gapPenalty;
        std::cin >> k;
        std::string genes[k];
        for (int i = 0; i < k; i++)
            std::cin >> genes[i];

        int numPairs = k * (k - 1) / 2;

        int penalties[numPairs];

        uint64_t start = GetTimeStamp();

        // return all the penalties and the hash of all allignments
        std::string alignmentHash = getMinimumPenalties(genes,
                                                        k, misMatchPenalty, gapPenalty,
                                                        penalties);

        // print the time taken to do the computation
        printf("Time: %ld us\n", (uint64_t)(GetTimeStamp() - start));

        // print the alginment hash
        std::cout << alignmentHash << std::endl;

        for (int i = 0; i < numPairs; i++)
        {
            std::cout << penalties[i] << " ";
        }
        std::cout << std::endl;
    }
    else
    {
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
// #include <tuple>
#include <algorithm>
#include <queue>

constexpr int SHA512_STRLEN = 129; // 128+1 for '\0'
constexpr int NEW_TASK_FLAG = 0;
constexpr int COLLECT_RESULT_TAG = 1;
constexpr int NO_MORE_TASK = -1;
int n_threads = 16;

// define the type for a job
typedef struct
{
    int i, j, id;
} JOB_t;

// define the type for a result
typedef struct
{
    int penalty, id;
    char problemhash[SHA512_STRLEN];
} RESULT_t;

inline MPI_Datatype create_MPI_JOB()
{
    // define the job type for MPI
    MPI_Datatype MPI_JOB;
    MPI_Type_contiguous(3, MPI_INT, &MPI_JOB);
    MPI_Type_commit(&MPI_JOB);
    return MPI_JOB;
}

inline MPI_Datatype create_MPI_RESULT()
{
    MPI_Datatype MPI_RESULT;
    int blen[3];
    MPI_Aint array_of_displacements[3];
    MPI_Datatype oldtypes[3];
    blen[0] = 1;
    array_of_displacements[0] = offsetof(RESULT_t, penalty);
    oldtypes[0] = MPI_INT;
    blen[1] = 1;
    array_of_displacements[1] = offsetof(RESULT_t, id);
    oldtypes[1] = MPI_INT;
    blen[2] = SHA512_STRLEN;
    array_of_displacements[2] = offsetof(RESULT_t, problemhash);
    oldtypes[2] = MPI_CHAR;
    MPI_Type_create_struct(3, blen, array_of_displacements, oldtypes, &MPI_RESULT);
    MPI_Type_commit(&MPI_RESULT);
    return MPI_RESULT;
}

inline RESULT_t do_job(std::string x, std::string y, int job_id, int misMatchPenalty, int gapPenalty);

inline bool compareByJobId(const RESULT_t &a, const RESULT_t &b)
{
    return a.id < b.id;
}

inline int min3(int a, int b, int c)
{
    if (a <= b && a <= c)
    {
        return a;
    }
    else if (b <= a && b <= c)
    {
        return b;
    }
    else
    {
        return c;
    }
}

// equivalent of  int *dp[width] = new int[height][width]
// but works for width not known at compile time.
// (Delete structure by  delete[] dp[0]; delete[] dp;)
inline int **new2d(int width, int height)
{
    int **dp = new int *[width];
    size_t size = width;
    size *= height;
    int *dp0 = new int[size];
    if (!dp || !dp0)
    {
        std::cerr << "getMinimumPenalty: new failed" << std::endl;
        exit(1);
    }
    dp[0] = dp0;
    for (int i = 1; i < width; i++)
        dp[i] = dp[i - 1] + height;

    return dp;
}

// called by the root MPI task only
// this procedure should distribute work to other MPI tasks
// and put together results, etc.
std::string getMinimumPenalties(std::string *genes, int k, int pxy, int pgap,
                                int *penalties)
{

    std::string alignmentHash = "";

    int size;
    MPI_Comm_size(comm, &size);

    // Broadcast the 3 configuration number
    int config[3] = {k, pxy, pgap}; // k, pxy, pgap
    MPI_Bcast(config, 3, MPI_INT, root, comm);

    // Broadcast the sequence length list
    int seq_length[k];

    for (int i = 0; i < k; ++i)
    {
        seq_length[i] = genes[i].length();
    }
    MPI_Bcast(seq_length, k, MPI_INT, root, comm);

    // Broadcast the sequences
    for (int i = 0; i < k; i++)
    {
        char buffer[seq_length[i]];
        memcpy(buffer, genes[i].c_str(), seq_length[i]);
        MPI_Bcast(buffer, seq_length[i], MPI_CHAR, root, comm);
    }

    // omp_set_nested(1);
    n_threads--;

    #pragma omp parallel default(shared) num_threads(2)
    {
        if (omp_get_thread_num() == 0) {
            // printf("I am %d from group %d\n",omp_get_thread_num(), omp_get_ancestor_thread_num(1));
            // create job id (#cell, <i, j, job-id>)
            queue<JOB_t> jobs;
            int job_id = 0;
            for (int i = 1; i < k; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    jobs.push({i, j, job_id++});

                    // jobs.push_back({(seq_length[i/10000])*(seq_length[j]/10000), {i, j, job_id++}});
                }
            }

            // send the initial job
            MPI_Datatype MPI_JOB = create_MPI_JOB();

            for (int i = 0; i < size; i++)
            {
                // get the job
                JOB_t job = jobs.front();
                jobs.pop();
                // send to worker i
                MPI_Send(&job, 1, MPI_JOB, i, NEW_TASK_FLAG, comm);
            }

            // keep a list of whether each worker is done
            vector<bool> n_done = {};
            int n_woker = size;

            // keep distributed the work
            MPI_Datatype MPI_RESULT = create_MPI_RESULT();
            MPI_Status status;
            vector<RESULT_t> results = {};

            while (n_done.size() < n_woker)
            {
                RESULT_t temp;
                MPI_Recv(&temp, 1, MPI_RESULT, MPI_ANY_SOURCE, COLLECT_RESULT_TAG, comm, &status);
                // cout << "rank-" << status.MPI_SOURCE << "result:" << temp.penalty << " " << temp.id << " " << temp.problemhash << endl;
                
                results.push_back(temp);
                // if there are still jobs to do
                if (!jobs.empty())
                {
                    // get the job
                    JOB_t job = jobs.front();
                    jobs.pop();
                    // send to worker i
                    MPI_Send(&job, 1, MPI_JOB, status.MPI_SOURCE, NEW_TASK_FLAG, comm);
                    // if there are nothing to do
                }
                else
                {
                    // ask the worker to stop
                    JOB_t job = {NO_MORE_TASK, NO_MORE_TASK, NO_MORE_TASK};
                    MPI_Send(&job, 1, MPI_JOB, status.MPI_SOURCE, NEW_TASK_FLAG, comm);
                    n_done.push_back(true);
                }
            }

            // std::cout << "while end" << endl;

            std::sort(results.begin(), results.end(), compareByJobId);

            for (int i = 0; i < results.size(); ++i)
            {
                alignmentHash = sw::sha512::calculate(alignmentHash.append(results[i].problemhash));
                penalties[i] = results[i].penalty;
            }

            // std::cout << "hash end" << endl;
        } else {
            // printf("I am %d from group %d\n",omp_get_thread_num(), omp_get_ancestor_thread_num(1));
            MPI_Status status;
            // receive my initial job
            JOB_t my_job;
            MPI_Datatype MPI_JOB = create_MPI_JOB();
            MPI_Recv(&my_job, 1, MPI_JOB, root, NEW_TASK_FLAG, comm, &status);
            // cout << "rank-" << rank << ": i=" << my_job.i << ", j=" << my_job.j << ", job-id=" << my_job.id << endl;
            int STOP = my_job.i;

            while (STOP != NO_MORE_TASK)
            {
                RESULT_t result = do_job(genes[my_job.i], genes[my_job.j], my_job.id, pxy, pgap);
                MPI_Datatype MPI_RESULT = create_MPI_RESULT();
                MPI_Send(&result, 1, MPI_RESULT, root, COLLECT_RESULT_TAG, comm);
                MPI_Recv(&my_job, 1, MPI_JOB, root, NEW_TASK_FLAG, comm, &status);
                // cout << "rank-" << rank << ": i=" << my_job.i << ", j=" << my_job.j << ", job-id=" << my_job.id << endl;
                STOP = my_job.i;
            }

            // std::cout << "all job done" << endl;
        }

        // std::cout << "pargma single end" << endl;
    }

    return alignmentHash;
}

// called for all tasks with rank!=root
// do stuff for each MPI task based on rank
void do_MPI_task(int rank)
{
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
    for (int i = 0; i < k; i++)
    {
        char buffer[seq_length[i] + 1];
        MPI_Bcast(buffer, seq_length[i], MPI_CHAR, root, comm);
        buffer[seq_length[i]] = '\0';
        genes[i] = string(buffer, seq_length[i]);
        // cout << "rank " << rank << "  " << genes[i] << endl;
    }

    // receive my initial job
    JOB_t my_job;
    MPI_Datatype MPI_JOB = create_MPI_JOB();
    MPI_Recv(&my_job, 1, MPI_JOB, root, NEW_TASK_FLAG, comm, &status);
    // cout << "rank-" << rank << ": i=" << my_job.i << ", j=" << my_job.j << ", job-id=" << my_job.id << endl;
    int STOP = my_job.i;

    while (STOP != NO_MORE_TASK)
    {
        RESULT_t result = do_job(genes[my_job.i], genes[my_job.j], my_job.id, misMatchPenalty, gapPenalty);
        MPI_Datatype MPI_RESULT = create_MPI_RESULT();
        MPI_Send(&result, 1, MPI_RESULT, root, COLLECT_RESULT_TAG, comm);

        MPI_Recv(&my_job, 1, MPI_JOB, root, NEW_TASK_FLAG, comm, &status);
        // cout << "rank-" << rank << ": i=" << my_job.i << ", j=" << my_job.j << ", job-id=" << my_job.id << endl;
        STOP = my_job.i;
    }
}

// function to find out the minimum penalty
// return the minimum penalty and put the aligned sequences in xans and yans
inline int getMinimumPenalty(std::string x, std::string y, int pxy, int pgap, int *xans, int *yans)
{

    int i, j; // intialising variables

    int m = x.length(); // length of gene1
    int n = y.length(); // length of gene2

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

inline RESULT_t do_job(std::string gene1, std::string gene2, int job_id, int misMatchPenalty, int gapPenalty)
{

    int m = gene1.length(); // length of gene1
    int n = gene2.length(); // length of gene2
    int l = m + n;
    int xans[l + 1], yans[l + 1];
    int penalty = getMinimumPenalty(gene1, gene2, misMatchPenalty, gapPenalty, xans, yans);

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

    RESULT_t result;
    result.penalty = penalty;
    result.id = job_id;
    strcpy(result.problemhash, problemhash.c_str());

    return result;
}
