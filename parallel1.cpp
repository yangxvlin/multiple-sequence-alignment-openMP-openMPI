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

int getMinimumPenalty2(std::string x, std::string y, int pxy, int pgap, int *xans, int *yans, int m, int n);

const int n_threads = 4;
const int sha512_strlen = 128 + 1; // +1 for '\0'

const int ask_for_genes_tag = 1;
const int send_genes_tag = 2;
const int collect_results_tag = 3;

struct Triple { 
   int x, y, z; 
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
    MPI_Status status;
    omp_set_num_threads(n_threads);
	int probNum=0;

    int size;
    MPI_Comm_size(comm, &size);

    // send k, pxy, pgap to wrokers
    const int k_pxy_pgap[3] = {k, pxy, pgap};
    // #pragma omp parallel for // this slows down program
    for (int i = 1; i < size; i++) {
        MPI_Send(&k_pxy_pgap, 3, MPI_INT, i, 1, comm);
    }

    int total = k * (k-1) / 2;
    // calculates string length
    int genes_length[k];
    for (int i = 0; i < k; i++) {
        genes_length[i] = genes[i].length();
    }

    for (int i = 1; i < size; i++) {
        int n_genes_to_sent;
        MPI_Recv(&n_genes_to_sent, 1, MPI_INT, i, ask_for_genes_tag, comm, &status);
        int genes_to_sent_id[n_genes_to_sent];
        MPI_Recv(genes_to_sent_id, n_genes_to_sent, MPI_INT, i, ask_for_genes_tag, comm, &status);
        // print genes to send
        cout << "rank[0][recv] rank[" << i << "][want] string(id): ";
        for (int j = 0; j < n_genes_to_sent; j++) {
            cout << genes_to_sent_id[j] << " ";
            MPI_Send(&(genes_length[genes_to_sent_id[j]]), 1, MPI_INT, i, j, MPI_COMM_WORLD);
            MPI_Send(genes[genes_to_sent_id[j]].c_str(), genes_length[genes_to_sent_id[j]], MPI_CHAR, i, j, MPI_COMM_WORLD);
        }
        cout << endl;
    }

    cout << "rank[0][tasks generation start]" << endl;

    // do root's tasks
    // number of dp matrix calculation per process
    int tasks_per_process = (int) floor((1.0*total) / size);
    int my_tasks_start = tasks_per_process * (size-1), my_tasks_end = total; // lask chunk of tasks on root
    // cout << "my_tasks_start " << my_tasks_start << endl;

    // ask root for the genes needed for calculation
    vector<Triple> tasks; // i, j, id of (i, j) in whole tasks
    int task_id = 0;
    for(int i=1;i<k;i++){
		for(int j=0;j<i;j++){
            if (task_id >= my_tasks_start && task_id < my_tasks_end) {
                tasks.push_back({ i, j, task_id });
            } else if (task_id >= my_tasks_end) {
                break;
            }
            task_id++;
        }
    }
    cout << "rank[0][tasks generation finish]" << endl;

    string answers_hash[total];
    // char answers_hash_array[total][sha512_strlen];
    int n_tasks = tasks.size();
    for (int z = 0; z < n_tasks; z++) {
        // cout << "rank[0] " << z << endl;
        Triple xyz = tasks.at(z);
        int i = xyz.x, j = xyz.y, task_id = xyz.z;
        int l = genes_length[i] + genes_length[j];
        int xans[l+1], yans[l+1];
        penalties[task_id] = getMinimumPenalty2(genes[i], genes[j], pxy, pgap, xans, yans, genes_length[i], genes_length[j]);
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
        // answers_hash_array[task_id] = problemhash.c_str();
        cout << "rank[0][calc] " << "task id: " << task_id << ", penalty: " << penalties[task_id] << ", hash: " << answers_hash[task_id] << endl;
    }

    cout << "rank[0][calc] finish" << endl;

    // build mpi Packet type
    MPI_Datatype MPI_Packet;
    int block_lengths[] = { 1, 1, 129 };
    // block_lengths[0] = sizeof(int);
    // block_lengths[1] = sizeof(int);
    // block_lengths[2] = strlen;
    MPI_Aint displacements[] = { 0, sizeof(int), sizeof(int) + sizeof(char) };
    MPI_Datatype types[] = { MPI_INT, MPI_INT, MPI_CHAR };
    MPI_Type_create_struct(3, block_lengths, displacements, types, &MPI_Packet);
    MPI_Type_commit(&MPI_Packet);
    // recv results form worker
    
    for (int i = 1; i < size; i++) {
        // cout << "rank[0][recv][start] answer from " << "rank: " << i << endl;
        Packet packest_recv[tasks_per_process];
        MPI_Recv(packest_recv, sizeof(packest_recv) * tasks_per_process, MPI_Packet, i, collect_results_tag, comm, &status);
        for (int j = 0; j < tasks_per_process; j++) {
            // cout << j << endl;
            penalties[packest_recv[j].task_id] = packest_recv[j].task_penalty;
            answers_hash[packest_recv[j].task_id] = string(packest_recv[j].task_hash, 128);
            cout << "id: " << packest_recv[j].task_id << ", " << packest_recv[j].task_hash << endl;
            // copy(packest_recv[j].task_hash, packest_recv[j].task_hash + 128,std::back_inserter(answers_hash[packest_recv[j].task_id]));
            // answers_hash[packest_recv[j].task_id] = string(packest_recv[j].task_hash, 128);
            // answers_hash_array[packest_recv[j].task_id] = packest_recv[j].task_hash;
            // strncpy ( answers_hash_array[packest_recv[j].task_id], packest_recv[j].task_hash, sha512_strlen );
            // cout << j << endl;
        }
        cout << "rank[0][recv] answer from " << "rank: " << i << endl;
    }

    std::string alignmentHash="";
    for (int i = 0; i < total; i++) {
        // if (i < my_tasks_start) {
        //     cout << "< " << alignmentHash << endl;
        //     cout << ">("<< strlen(answers_hash_array[i]) <<") " << answers_hash_array[i] << endl;
        //     alignmentHash = sw::sha512::calculate(alignmentHash.append(string(answers_hash_array[i], 128)));
        //     cout << alignmentHash << endl;
        //     std::cout << std::endl;
        //     continue;
        // }

        // aggregrate result
        cout << "< " << alignmentHash << endl;
        cout << ">("<< answers_hash[i].size() <<") " << answers_hash[i] << endl;
        alignmentHash = sw::sha512::calculate(alignmentHash.append(answers_hash[i]));
        cout << alignmentHash << endl;
        std::cout << std::endl;
    }

	return alignmentHash;
}

// called for all tasks with rank!=root
// do stuff for each MPI task based on rank
void do_MPI_task(int rank) {
    omp_set_num_threads(n_threads);
    MPI_Status status;
    int size;
    MPI_Comm_size(comm, &size);

    int k_pxy_pgap[3];
    MPI_Recv(&k_pxy_pgap, 3, MPI_INT, root, 1, comm, &status);
    int k = k_pxy_pgap[0], 
        pxy = k_pxy_pgap[1],
        pgap = k_pxy_pgap[2];

    cout << "rank[" << rank << "][recv] " << "k: " << k << ", pxy: " << pxy << ", pgap: " << pgap << endl;

    int total = k * (k-1) / 2;

    // number of dp matrix calculation per process
    int tasks_per_process = (int) floor((1.0*total) / size);
    int my_tasks_start = tasks_per_process * (rank-1), my_tasks_end = tasks_per_process * rank;

    // ask root for the genes needed for calculation
    vector<Triple> tasks; // i, j, id of (i, j) in whole tasks
    unordered_set<int> genes_to_ask;
    // calculate my tasks to do;
    int task_id = 0;
    for(int i=1;i<k;i++){
		for(int j=0;j<i;j++){
            if (task_id >= my_tasks_start && task_id < my_tasks_end) {
                tasks.push_back({ i, j, task_id });
                genes_to_ask.insert(i);
                genes_to_ask.insert(j);
            } else if (task_id >= my_tasks_end) {
                break;
            }
            task_id++;
        }
    }
    // send my request
    vector<int> sorted_genes_to_ask;
    sorted_genes_to_ask.assign( genes_to_ask.begin(), genes_to_ask.end() );
    sort( sorted_genes_to_ask.begin(), sorted_genes_to_ask.end() );
    int* sorted_genes_to_ask_array = sorted_genes_to_ask.data();
    int n_genes = sorted_genes_to_ask.size();
    // cout << rank << " " << n_genes << endl;
    MPI_Send(&n_genes, 1, MPI_INT, root, ask_for_genes_tag, comm);
    MPI_Send(sorted_genes_to_ask_array, n_genes, MPI_INT, root, ask_for_genes_tag, comm);

    string local_genes[k];
    int local_genes_len[k];
    for (int i = 0; i < n_genes; i++) {
        // sorted_genes_to_ask_array[i]
        int string_len;
        MPI_Recv(&string_len, 1, MPI_INT, root, sorted_genes_to_ask_array[i], comm, &status);
        cout << "rank[" << rank << "][recv] " << "string id: " << sorted_genes_to_ask_array[i] << ", len: " << string_len << endl;
        char string_bufffer[string_len+1]; // +1 for string end
        MPI_Recv(string_bufffer, string_len, MPI_CHAR, root, sorted_genes_to_ask_array[i], MPI_COMM_WORLD, &status);
        string_bufffer[string_len] = '\0';
        cout << "rank[" << rank << "][recv] " << "string id: " << sorted_genes_to_ask_array[i] << ", len: " << string_len << ", string: " << string_bufffer << endl;

        local_genes[sorted_genes_to_ask_array[i]] = string_bufffer;
        local_genes_len[sorted_genes_to_ask_array[i]] = string_len;
    }

    // build mpi Packet type
    MPI_Datatype MPI_Packet;
    int block_lengths[] = { 1, 1, 129 };
    // block_lengths[0] = sizeof(int);
    // block_lengths[1] = sizeof(int);
    // block_lengths[2] = strlen;
    MPI_Aint displacements[] = { 0, sizeof(int), sizeof(int) + sizeof(char) };
    MPI_Datatype types[] = { MPI_INT, MPI_INT, MPI_CHAR };
    MPI_Type_create_struct(3, block_lengths, displacements, types, &MPI_Packet);
    MPI_Type_commit(&MPI_Packet);

    Packet packets[tasks_per_process];
    // do sequence alignment calculation
    for (int z = 0; z < tasks_per_process; z++) {
        Triple xyz = tasks.at(z);
        int i = xyz.x, j = xyz.y;
        packets[z].task_id = xyz.z;
        int l = local_genes_len[i] + local_genes_len[j];
        int xans[l+1], yans[l+1];
        packets[z].task_penalty = getMinimumPenalty2(local_genes[i], local_genes[j], pxy, pgap, xans, yans, local_genes_len[i], local_genes_len[j]);
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
        // cout << "rank[" << rank << "][calc] " << "task id: " << packets[z].task_id << ", align1: " << align1 << ", align2: " << align2 << endl;
        // cout << "rank[" << rank << "][calc] " << "task id: " << packets[z].task_id << ", align1hash: " << align1hash << ", align2hash: " << align2hash << endl;
        // store problemhash sent to root
        // packets[z].task_hash = problemhash.c_str();
        strcpy(packets[z].task_hash, problemhash.c_str()); 
        packets[z].task_hash[sha512_strlen] = '\0';
        cout << "rank[" << rank << "][calc] " << "task id: " << packets[z].task_id << ", penalty: " << packets[z].task_penalty << ", hash("<< strlen(packets[z].task_hash) <<"): " << packets[z].task_hash << endl;
    }    

    // sent jobs result to root
    MPI_Send(packets, tasks_per_process, MPI_Packet, root, collect_results_tag, comm);
    cout << "rank[" << rank << "][finish] " << endl;
}

int getMinimumPenalty2(std::string x, std::string y, int pxy, int pgap, int *xans, int *yans, int m, int n) {
	
	int i, j; // intialising variables

	// int m = x.length(); // length of gene1
	// int n = y.length(); // length of gene2
	
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
