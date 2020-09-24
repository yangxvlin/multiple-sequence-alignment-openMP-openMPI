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

constexpr int JOB_DISTRIBUTION_TAG {0};
constexpr int RESULT_COLLECTION_TAG = 1;
constexpr int STOP_SYMBOL = -9;
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
    char problemhash[129];
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
    blen[2] = 129;
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
                MPI_Send(&job, 1, MPI_JOB, i, JOB_DISTRIBUTION_TAG, comm);
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
                MPI_Recv(&temp, 1, MPI_RESULT, MPI_ANY_SOURCE, RESULT_COLLECTION_TAG, comm, &status);
                // cout << "rank-" << status.MPI_SOURCE << "result:" << temp.penalty << " " << temp.id << " " << temp.problemhash << endl;
                
                results.push_back(temp);
                // if there are still jobs to do
                if (!jobs.empty())
                {
                    // get the job
                    JOB_t job = jobs.front();
                    jobs.pop();
                    // send to worker i
                    MPI_Send(&job, 1, MPI_JOB, status.MPI_SOURCE, JOB_DISTRIBUTION_TAG, comm);
                    // if there are nothing to do
                }
                else
                {
                    // ask the worker to stop
                    JOB_t job = {STOP_SYMBOL, STOP_SYMBOL, STOP_SYMBOL};
                    MPI_Send(&job, 1, MPI_JOB, status.MPI_SOURCE, JOB_DISTRIBUTION_TAG, comm);
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
            MPI_Recv(&my_job, 1, MPI_JOB, root, JOB_DISTRIBUTION_TAG, comm, &status);
            // cout << "rank-" << rank << ": i=" << my_job.i << ", j=" << my_job.j << ", job-id=" << my_job.id << endl;
            int STOP = my_job.i;

            while (STOP != STOP_SYMBOL)
            {
                RESULT_t result = do_job(genes[my_job.i], genes[my_job.j], my_job.id, pxy, pgap);
                MPI_Datatype MPI_RESULT = create_MPI_RESULT();
                MPI_Send(&result, 1, MPI_RESULT, root, RESULT_COLLECTION_TAG, comm);
                MPI_Recv(&my_job, 1, MPI_JOB, root, JOB_DISTRIBUTION_TAG, comm, &status);
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
    MPI_Recv(&my_job, 1, MPI_JOB, root, JOB_DISTRIBUTION_TAG, comm, &status);
    // cout << "rank-" << rank << ": i=" << my_job.i << ", j=" << my_job.j << ", job-id=" << my_job.id << endl;
    int STOP = my_job.i;

    while (STOP != STOP_SYMBOL)
    {
        RESULT_t result = do_job(genes[my_job.i], genes[my_job.j], my_job.id, misMatchPenalty, gapPenalty);
        MPI_Datatype MPI_RESULT = create_MPI_RESULT();
        MPI_Send(&result, 1, MPI_RESULT, root, RESULT_COLLECTION_TAG, comm);

        MPI_Recv(&my_job, 1, MPI_JOB, root, JOB_DISTRIBUTION_TAG, comm, &status);
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

    // table for storing optimal substructure answers
    omp_set_num_threads(n_threads);
    int **dp = new2d(m + 1, n + 1);

    // remove unnecessary memset
    //    size_t size = m + 1;
    //    size *= n + 1;
    //    memset (dp[0], 0, size);

    // intialising the table
    #pragma omp parallel for
    for (i = 0; i <= m; ++i)
    {
        dp[i][0] = i * pgap;
    }
    #pragma omp parallel for
    for (i = 0; i <= n; ++i)
    {
        dp[0][i] = i * pgap;
    }

    // calculating the minimum penalty with the tiling technique in an anti-diagonal version
    int tile_row_size = (int)ceil((1.0 * m) / n_threads); // Number of dp elements in row of each tile
    int tile_col_size = (int)ceil((1.0 * n) / n_threads); // Number of dp elements in column of each tile

    //    int tile_row_size = 256; // Number of dp elements in row of each tile
    //    int tile_col_size = 256; // Number of dp elements in column of each tile
    int tile_m = (int)ceil((1.0 * m) / tile_row_size); // Number of tiles in row of the dp matrix
    int tile_n = (int)ceil((1.0 * n) / tile_col_size); // Number of tile in column of the dp matrix

    int total_diagonal = tile_m + tile_n - 1;
    int row_min, row_max, diagonal_index, k;
    //    cout << "tile_row_size: " << tile_row_size << ", tile_col_size: " << tile_col_size << endl;
    //    cout << "tile_m: " << tile_m << ", tile_n: " << tile_n << endl;
    //    cout << "total_diagonal: " << total_diagonal << endl;
    for (diagonal_index = 1; diagonal_index <= total_diagonal; ++diagonal_index)
    {
        row_min = max(1, diagonal_index - tile_n + 1);
        row_max = min(diagonal_index, tile_m);

        #pragma omp parallel for
        for (k = row_min; k <= row_max; ++k)
        {
            int tile_row_start = 1 + (k - 1) * tile_row_size;              // index inclusive
            int tile_row_end = min(tile_row_start + tile_row_size, m + 1); // index exclusive
            int tile_col_start = 1 + (diagonal_index - k) * tile_col_size; // index inclusive
            int tile_col_end = min(tile_col_start + tile_col_size, n + 1); // index exclusive

            //            cout << "(" << tile_row_start<< "," << tile_col_start << ")" << " | ";
            //            cout << "-> (" << tile_row_end << "," << tile_col_end << ")" << '|';
            for (int ii = tile_row_start; ii < tile_row_end; ++ii)
            {
                for (int jj = tile_col_start; jj < tile_col_end; ++jj)
                {
                    if (x[ii - 1] == y[jj - 1])
                    {
                        dp[ii][jj] = dp[ii - 1][jj - 1];
                    }
                    else
                    {
                        dp[ii][jj] = min3(dp[ii - 1][jj - 1] + pxy,
                                          dp[ii - 1][jj] + pgap,
                                          dp[ii][jj - 1] + pgap);
                    }
                }
            }
        }
        //        cout << "n_done" << endl;
    }

    // Reconstructing the solution
    int l = n + m; // maximum possible length

    i = m;
    j = n;

    int xpos = l;
    int ypos = l;

    while (!(i == 0 || j == 0))
    {
        if (x[i - 1] == y[j - 1])
        {
            xans[xpos--] = (int)x[i - 1];
            yans[ypos--] = (int)y[j - 1];
            i--;
            j--;
        }
        else if (dp[i - 1][j - 1] + pxy == dp[i][j])
        {
            xans[xpos--] = (int)x[i - 1];
            yans[ypos--] = (int)y[j - 1];
            i--;
            j--;
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
    int x_diff = xpos - i, y_diff = ypos - j;
    #pragma omp parallel for
    for (int ii = i; ii > 0; --ii)
    {
        xans[ii + x_diff] = (int)x[ii - 1];
    }

    #pragma omp parallel for
    for (int x_pos2 = xpos - i; x_pos2 > 0; --x_pos2)
    {
        xans[x_pos2] = (int)'_';
    }

    #pragma omp parallel for
    for (int jj = j; jj > 0; --jj)
    {
        yans[jj + y_diff] = (int)y[jj - 1];
        if (jj == 0)
        {
        }
    }

    #pragma omp parallel for
    for (int y_pos2 = ypos - j; y_pos2 > 0; --y_pos2)
    {
        yans[y_pos2] = (int)'_';
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
