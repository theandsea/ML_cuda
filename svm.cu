using namespace std;
#include<iostream>
#include<stdio.h>
#include<fstream>
#include <math.h>
#include <string>
#include <time.h>
clock_t start_t, end_t;
double cpu_time, gpu_time; 

// only in c++
#define new_max(x,y) (((x) >= (y)) ? (x) : (y))

#define img_size 28*28//28*28
#define class_num 10
#define train_size 1000 // must larger than 100, other than no 6
#define test_size 1000 // for validation to ge the best w, b
#define valid_size 1000 // for validation to ge the best w, b
#define max_size new_max(new_max(train_size,valid_size),test_size)
#define offset 0

#define BDIMY 32
#define BDIMX 32
#define b_len 256
#define train_block (train_size+b_len-1)/b_len
#define valid_block (valid_size+b_len-1)/b_len
#define test_block (test_size+b_len-1)/b_len


static int magic_number = 0;
static int number_of_images = 0;
static int n_rows = 0;
static int n_cols = 0;
static float test_x[test_size][img_size];
static float train_x[train_size][img_size];
static float valid_x[valid_size][img_size]; 

static int test_t[test_size]; //[28*28] wrong size...cause core dump
static int train_t[train_size]; //[28*28] wrong size...cause core dump
static int valid_t[valid_size]; //[28*28] wrong size...cause core dump

static int test__y[test_size];
static int train__y[train_size]; 
static int valid__y[valid_size];

// for save
static float w_s[class_num][img_size];
static float b_s[class_num]; // for cpu
static float b_gpu[class_num]; // for gpu

float *d_test_x, *d_train_x, *d_valid_x, *d_kernel;
float *d_test_x_trans, *d_train_x_trans, *d_valid_x_trans;
int *d_test_t, *d_train_t, *d_valid_t;
int* d_test__y[class_num], *d_train__y[class_num], *d_valid__y[class_num];
float *d_a[class_num], *d_w[class_num], *d_best_w[class_num];
float best_error_[class_num];//,b_[class_num];
// for temp use
float *d_ayk[class_num],*d_ay[class_num], *d_y_est[class_num], *d_c_est; // temp
float *d_y_est_forc;

// f
int svm(float img[]);
int svm_general(float img[]);
void w_update(float img[][img_size]);
float svm_loss();
void arr_cp(float x_copy[], float x[], int l);
float error_rate_general(float img[][img_size], int label[], int size);


// for define gpu function...invoke cuda
float gpu_d1_3_multi(float* a, int* y, float* kernel, int l, int c);
void check_difference(float *d, float *h, int is,int js);
float gpu_error_rate(float* w,float* img_t, int* y, float b, int ts , int c);
float gpu_svm_loss(int c);
float gpu_d1_sum(float* x, const int ts);
float gpu_d1_sum_nonvolitile(float* x, const int ts);


// cuda 
__global__ void d_d1_sum(float* x, float* sum, const int ts);
__global__ void d_d1_3_multi(float* a, int* y, float* kernel, float* res, const int ts);
__global__ void d_d1_2_multi(float* a, int* y, float* res, const int ts);
__global__ void d_d1_2_multi(float* a, float* y, float* res, const int ts);
__global__ void d_d1_d2(float *w, float *img_t, float *y_est,const float b, const int ws, const int ts);
__global__ void d_transpose(float *out, float *in, int nx, int ny);
__global__ void d_t_y(int* t, int* y, const int c, const int ts);
__global__ void d_d1_d1_y_comp(int *y, float *y_est, const int ts);
__global__ void d_d2_maxidx_d1_comp(float *d_y_est, int* d_t, const int ts);



// for read the data
// this part can't be parallelized
//
int reverseInt(int i){
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
void read_mnist(string path_x, string path_t, float img[][img_size], int label[], int size)  // at least 2nd must fixed size
// (/*string full_path*/)
{
    ifstream file(path_x);//, ios::out | ios::binary //("t10k-images-idx3-ubyte.gz");// can use .gz...still need unzip !!!
    if (file.is_open())
    {
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);

        // read image
        printf("read image\n");
        printf("%d    %d   %d   %d\n", magic_number, number_of_images, n_rows, n_cols);

        // read data
        unsigned char temp;
        float normalized = 255.0;

        // shift data
        if (number_of_images >10010) // training date
        for (int i = 0; i < offset; ++i)
            for (int r = 0; r < n_rows; ++r)
                for (int c = 0; c < n_cols; ++c) {
                    file.read((char*)&temp, 1);
                }

        for (int i = 0; i < min(number_of_images, size); ++i) 
        // !!!!!!!!!!!!!!!
        // can't (train_size > number_of_image)..thus error...thus for test size, it should be at 
            for (int r = 0; r < n_rows; ++r)
                for (int c = 0; c < n_cols; ++c) {
                    file.read((char*)&temp, 1);
                    // file.read((char*)&(img[i][r*n_cols+c]),sizeof(unsigned char));
                    img[i][r * n_cols + c] = temp / normalized;// * 2.0 -1.0;
                }
        
        if (number_of_images >10010){ // training date
            for (int i = 0; i < valid_size; ++i)
                for (int r = 0; r < n_rows; ++r)
                    for (int c = 0; c < n_cols; ++c) {
                        file.read((char*)&temp, 1);
                        // file.read((char*)&(img[i][r*n_cols+c]),sizeof(unsigned char));
                        valid_x[i][r * n_cols + c] = temp / normalized;// * 2.0 -1.0;
                    }
            /*
            for (int i = 0; i < valid_size; ++i)
                for (int r = 0; r < n_rows; ++r)
                    for (int c = 0; c < n_cols; ++c) {
                        file.read((char*)&temp, 1);
                        // file.read((char*)&(img[i][r*n_cols+c]),sizeof(unsigned char));
                        test_x[i][r * n_cols + c] = temp / normalized;// * 2.0 -1.0;
                    }
            */
        }
    }
    file.close();


    // ifstream file(path_t);
    file.open(path_t);
    if (file.is_open())
    {
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);

        printf("read label\n");
        printf("%d    %d\n", magic_number, number_of_images);

        // read data
        if (number_of_images >10010) // training date
        for (int i = 0; i < offset; ++i) {
            file.read((char*)&(label[i]), sizeof(char)); 
        }

        for (int i = 0; i < min(number_of_images, size); ++i) {
            file.read((char*)&(label[i]), sizeof(char)); //
            //if (i<10)
            // printf("%d    %d   %d\n",i,label[i],number_of_images);
        }

        int zeros=0;
        if (number_of_images >10010){ // training date
            for (int i = 0; i < valid_size; ++i) {
                file.read((char*)&(valid_t[i]), sizeof(char));
                if (valid_t[i] == 0)
                    zeros ++;
            }
            /*
            for (int i = 0; i < test_size; ++i) {
                file.read((char*)&(test_t[i]), sizeof(char));
            }
            */
        }
        printf("zeros = %d\n",zeros);
        
    }
    file.close();
}



void print_img(float img[img_size]) { // no need
    int i, j;
    for (i = 0; i < 28; i++) {
        for (j = 0; j < 28; j++)
            if (img[i * 28 + j] > 0.0f) // otherwise, 0.4 > 0 is not true
                printf("1 ");
            else
                printf("  ");
        printf("\n");
    }
}

// convert class to -1,1

void t_y(int label[], int y[], int c, int l) {
    int i;
    for (i = 0; i < l; i++) {
        //printf("label[]: %d\n", label[i]);
        if (label[i] == c) {
            y[i] = 1;
            // printf("i :%d\n", i);
        }
        else {
            y[i] = -1;
        }
    }
}



float error_rate(float img[][img_size], int label[], int size, int  (*func)(float img[])) { // merged with svm
    // must type in (*func)(int img[])
    // int size=sizeof(label)/sizeof(int);
    int i;
    int error = 0;
    int compare = 0;
    
    for (i = 0; i < size; i++) {
        // printf("i=%d    ",i);
        if ((*func)(img[i]) != label[i]) error++;
        // if (label[i] != 1) compare ++;

        // binary
        /*
        est =(*func)(img[i]);

        if ((est==1 && label[i]!=c)|| (est==-1 && label[i]==c) )
            error++;
        */

    }
    printf("------test for error rate-----%d   %d  %d\n", size, error, compare);
    return ((float)error) / size;
}


float error_rate_general(float img[][img_size], int label[], int size) { // merged with svm
    // must type in (*func)(int img[])
    // int size=sizeof(label)/sizeof(int);
    int i;
    int error = 0;
    
    for (i = 0; i < size; i++) {
        // comprehensive
        // printf("i=%d    est=%d     label=%d\n",i,svm_general(img[i]),label[i]);
        if(svm_general(img[i]) != label[i])
            error++;
    }
    printf("------test for comprehensive error rate-----%d   %d\n", size, error);
    return ((float)error) / size;
}


float gpu_error_rate(float* w,float* img_t, int* y, float b, int ts , int c){
    d_d1_d2<<<(ts+b_len-1)/b_len,b_len>>>(w, img_t, d_y_est[c],b, img_size, ts);
    d_d1_d1_y_comp<<<(ts+b_len-1)/b_len,b_len>>>(y, d_y_est[c], ts);

    float sum = gpu_d1_sum(d_y_est[c], ts);
    // printf("sum = %f   ts= %d\n",sum,ts);

    return 1-sum/ts;
}

float gpu_error_rate_general(float* img_t, int* t, int ts) { // merged with svm

    // check_difference(d_w[0], w_s[0], 1,img_size);

    int c;
    for(c=0;c<class_num;c++)
        d_d1_d2<<<(ts+b_len-1)/b_len,b_len>>>(d_w[c], img_t, &d_y_est_forc[c*ts],b_s[c], img_size, ts);
    d_d2_maxidx_d1_comp<<<(ts+b_len-1)/b_len,b_len>>>(d_y_est_forc, t, ts);

    float sum = gpu_d1_sum(d_y_est_forc, ts);

    return 1-sum/ts;
}

__global__ void d_d2_maxidx_d1_comp(float *d_y_est_forc, int* d_t, const int ts){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    
    if(idx < ts){
        // float* this_d_y_est = d_y_est[0];
        float max_y = d_y_est_forc[idx];//= *(d_y_est[2]+idx);
        // printf("idx=%d  ts=%d   max_y=%f\n",idx,ts,max_y);
        int max_c = 0;
        int c;
        for (c=1;c<class_num;c++)
            if(max_y < d_y_est_forc[c*ts+idx]){
                max_y = d_y_est_forc[c*ts+idx];
                max_c = c;
            }
        // printf("idx=%d    est=%d     label=%d\n",idx,max_c,d_t[idx]);
        if (max_c == d_t[idx]){
            d_y_est_forc[idx] = 1.0f;
        } else {
            d_y_est_forc[idx] = 0.0f;
        }
    }
}


// training__binary
static float w[img_size];
static float b;
static float a[train_size];
static float C;
static float kernel[train_size][train_size];
static float best_error;
static float best_w[img_size];
// overall initialize
void overall_init(){
    int i,j,k;
    // kernel
    float res_each = 0;
    // int x1[img_size]; // not allow incomple
    // int x2[img_size];
    for (i = 0; i < train_size; i++) {
        for (j = 0; j <= i; j++) {  // similar to update 
            res_each = 0;
            for (k = 0; k < img_size; k++)
                res_each += train_x[i][k] * train_x[j][k];
            kernel[i][j] = res_each;
            kernel[j][i] = res_each;
        }
        // printf("i=%d\n",i);
    }
}
// initialize
void svm_init(float myc, float img[][img_size], int label[], int c) {
    // label-->y
    int l = train_size;
    //printf("label0 : %d\n", label[0]);
    t_y(label, train__y, c, l);
    t_y(valid_t, valid__y, c, valid_size);
    t_y(test_t, test__y, c, test_size);
    // C
    C = myc;
    // a = 0
    int i;
    // float sum = 0;
    for (i = 0; i < l - 1; i++) {
        a[i] = 0.0;//0.1;
        // sum += a[i] * train__y[i];
    } // just slow, but can't change b....not compatible
    a[l - 1] = 0.0; //-sum*train__y[l-1];
    b = 0;
    best_error = 1.0;

    // w
    for (i = 0; i < img_size; i++)
        w[i] = 0.0;
}






// svm loss checking
float svm_loss() { // no need, just for test
    int i, j;
    float res = 0;
    // kernel
    for (i = 0; i < train_size; i++)
        for (j = 0; j < train_size; j++)
            res += a[i] * a[j] * train__y[i] * train__y[j] * kernel[i][j];
    res *= -0.5;
    // a
    for (i = 0; i < train_size; i++)
        res += a[i];
    return (res);
}


void arr_cp(float x_copy[], float x[], int l){
    // copy and save 1d array for later use
    int i;
    for(i=0;i<l;i++)
        x_copy[i]=x[i];
}




// iteration
void svm_SMO_new(float img[][img_size], int y[]) {

    int j1, j2;
    int i;
    float prev = 0;
    float b_old;
    double error_rate_acc = 0.0;
    float error_rate_single;

    int iter; b=0.0;
    // j1 = (int)(static_cast <float> (rand()) / static_cast <float> (RAND_MAX ) * train_size); // +1
    // j2 = (int)(static_cast <float> (rand()) / static_cast <float> (RAND_MAX ) * train_size); // +1
    j1=0;
    j2=0;
    
    for (iter = 0; iter < 10 * train_size; iter++) {
        
        j1 = (j1 + 1) % train_size; // iteration can't be be parallel
        j2 = (j2 + 1) % train_size;
        while (y[j1] == -1) {
            j1 = (j1+1) % train_size;
        }
        while (y[j2] == 1) {
            j2 = (j2+1) % train_size;
        }

        int y1 = (int)y[j1];
        int y2 = (int)y[j2];
        float a1_old = a[j1];
        float a2_old = a[j2];
        float E1 = b - y1;
        for (i = 0; i < train_size; i++)
            E1 += a[i] * y[i] * kernel[j1][i];
        float E2 = b - y2;
        for (i = 0; i < train_size; i++)
            E2 += a[i] * y[i] * kernel[j2][i];
        float eta = kernel[j1][j1] - 2 * kernel[j1][j2] + kernel[j2][j2];

        float a2_new = a2_old + y2 * (E1 - E2) / eta;

        float L, H;
        float zero = 0.0;
        if (y1 != y2) {
            L = max(zero, a2_old - a1_old);
            H = min(C, C + a2_old - a1_old);
        }
        else {
            L = max(zero, a2_old + a1_old - C);
            H = min(C, a2_old + a1_old);
        }

        float a2_new_clip = a2_new;
        if (a2_new > H) {
            a2_new_clip = H;
        }
        else if (a2_new < L) {
            a2_new_clip = L;
        }

        float a1_new = a1_old + y1 * y2 * (a2_old - a2_new_clip);

        /**/
        float b1_new = b - E1 - y1 * kernel[j1][j1] * (a1_new - a1_old) - y2 * kernel[j1][j2] * (a2_new_clip - a2_old);
        float b2_new = b - E2 - y1 * kernel[j1][j2] * (a1_new - a1_old) - y2 * kernel[j2][j2] * (a2_new_clip - a2_old);
        float bnew = 0;
       if (0 < a1_new && a1_new < C) {
            printf("b1_new\n");
            bnew = b1_new;
        }
        else if (0 < a2_new_clip && a2_new_clip < C) {
            printf("b2_new\n");
            bnew = b2_new;
        }
        else {
            printf("(b1_new + b2_new)/2 \n");
            //if (abs(b1_new - b) > abs(b2_new - b))
            //    bnew = b2_new;
            //else if (abs(b1_new - b) < abs(b2_new - b))
            //    bnew = b1_new;
            //else
                bnew = b1_new *0.5 + b2_new * 0.5;
        }
       
        b_old = b;
        // update
        a[j1] = a1_new;
        a[j2] = a2_new_clip;
        w_update(img);
        b = bnew;

        printf("=====================\n");
        printf("choose  %d   %d\n", j1, j2);
        // error_rate_single = error_rate(train_x, y, train_size, svm);
        error_rate_single = error_rate(valid_x, valid__y, valid_size, svm); // by validation
        if (error_rate_single < best_error){
            best_error = error_rate_single;
            for(i=0;i<img_size;i++)
                best_w[i] = w[i];
        }
        printf("error rate for svm = %f\n", error_rate_single);
        float curr_loss = svm_loss();
        printf("loss(to max) = %f     deta = %f\n", curr_loss, curr_loss - prev);
        if (isnan(curr_loss) || iter > 0 && prev > curr_loss + 0.00001f) {
            // throw exception("algorithm wrong !!!\n");
            printf("very wrong !!!  %d \n", iter);
            // exit(1);
        }
        prev = curr_loss;
        //myfile << to_string(error_rate_single) << endl;
        error_rate_acc += error_rate_single / (10*train_size);
        

    }
    printf("error_rate_acc: %lf\n", error_rate_acc);
    //myfile.close();
}



// iteration
void svm_SMO_cuda(int c,float img[][img_size], int y[]) {

    int j1, j2;
    float prev = 0;
    double error_rate_acc = 0.0;
    float error_rate_single;

    int iter; 
    float b=0.0; // b=0; not same !!
    j1 =0;
    j2 =0;
    for (iter = 0; iter < 10 * train_size; iter++) {

        j1 = (j1 + 1) % train_size; // iteration can't be be parallel
        j2 = (j2 + 1) % train_size;
        while (y[j1] == -1) {
            j1 = (j1+1) % train_size;
        }
        while (y[j2] == 1) {
            j2 = (j2+1) % train_size;
        }

        int y1 = (int)y[j1];
        int y2 = (int)y[j2];



        float a1_old, a2_old;
        // cudaMemcpy(d_a[c]+j1,&a[j1],sizeof(float)*1, cudaMemcpyHostToDevice);
        // a1_old = a[j1];
        // a2_old = a[j2];
        cudaMemcpy(&a1_old, d_a[c]+j1, sizeof(float) ,cudaMemcpyDeviceToHost);
        cudaMemcpy(&a2_old, d_a[c]+j2, sizeof(float) ,cudaMemcpyDeviceToHost);

        /*
        float E1 = b - y1;
        for (i = 0; i < train_size; i++)
            E1 += a[i] * y[i] * kernel[j1][i];
        float E2 = b - y2;
        for (i = 0; i < train_size; i++)
            E2 += a[i] * y[i] * kernel[j2][i];
        */
        
        float d_E1 = b - y1 + gpu_d1_3_multi(d_a[c], d_train__y[c], &d_kernel[j1*train_size], train_size,c);
        float d_E2 = b - y2 + gpu_d1_3_multi(d_a[c], d_train__y[c], &d_kernel[j2*train_size], train_size,c);
        float E1=d_E1, E2=d_E2;


        if(abs(d_E1-E1) > 1e-3){
            printf("something is wrong!\n");

            printf("different result for E1 !!!  %d \n", iter);
            printf("E1=%f    d_E1=%f    difference=%f\n",E1,d_E1, E1-d_E1);
            exit(1);
        }
        printf("no difference for E1  iter=%d    E1=%d   d_E1=%d\n", iter, E1, d_E1);



        float eta = kernel[j1][j1] - 2 * kernel[j1][j2] + kernel[j2][j2];

        float a2_new = a2_old + y2 * (E1 - E2) / eta;

        float L, H;
        float zero = 0.0;
        if (y1 != y2) {
            L = max(zero, a2_old - a1_old);
            H = min(C, C + a2_old - a1_old);
        }
        else {
            L = max(zero, a2_old + a1_old - C);
            H = min(C, a2_old + a1_old);
        }

        float a2_new_clip = a2_new;
        if (a2_new > H) {
            a2_new_clip = H;
        }
        else if (a2_new < L) {
            a2_new_clip = L;
        }

        float a1_new = a1_old + y1 * y2 * (a2_old - a2_new_clip);

        /**/
        float b1_new = b - E1 - y1 * kernel[j1][j1] * (a1_new - a1_old) - y2 * kernel[j1][j2] * (a2_new_clip - a2_old);
        float b2_new = b - E2 - y1 * kernel[j1][j2] * (a1_new - a1_old) - y2 * kernel[j2][j2] * (a2_new_clip - a2_old);
        float bnew = 0;
       if (0 < a1_new && a1_new < C) {
            printf("b1_new\n");
            bnew = b1_new;
        }
        else if (0 < a2_new_clip && a2_new_clip < C) {
            printf("b2_new\n");
            bnew = b2_new;
        }
        else {
            printf("(b1_new + b2_new)/2 \n");
                bnew = b1_new *0.5 + b2_new * 0.5;
        }
       
        // update
        a[j1] = a1_new;
        a[j2] = a2_new_clip;
        // w_update(img);

        // update into cuda
        cudaMemcpy(d_a[c]+j1,&a[j1],sizeof(float)*1, cudaMemcpyHostToDevice);
        cudaMemcpy(d_a[c]+j2,&a[j2],sizeof(float)*1, cudaMemcpyHostToDevice);
        // printf("checking d_a vs a....\n");
        // check_difference(d_a[c],a,1,train_size);

        // d_d1_d2<<<(img_size+b_len-1)/b_len,b_len>>>(d_a[c],*train_x,d_w[c],0.0,train_size,img_size); // update 
        // this step make d_a corrupted ! not work !!!!!!!!!!!!!!!!!!!!!   *train_x is in device, make it work not properly, cuda not work any more(in check_difference) !!!! 
        d_d1_2_multi<<<(train_size+b_len-1)/b_len,b_len>>>(d_a[c],d_train__y[c],d_ay[c],train_size);
        // __global__ void d_d1_2_multi(float* a, float* y, float* res, const int ts)

        d_d1_d2<<<(img_size+b_len-1)/b_len,b_len>>>(d_ay[c],d_train_x,d_w[c],0.0f,train_size,img_size);
        // d_d1_d2<<<(train_size+b_len-1)/b_len,b_len>>>(&d_train_x[i*img_size],d_train_x_trans,&d_kernel[i*train_size],0.0f,img_size,train_size);
        // d_d1_d2(float *w, float *img_t, float *y_est,const float b, const int ws, const int ts)
        
        /// check_difference(d_kernel, *kernel, train_size,train_size);
        // printf("checking d_w vs w....\n");
        // check_difference(d_w[c], w,1,img_size);
        // void check_difference(float *d, float *h, int is,int js)



        b = bnew;

        /*
        // error_rate_single = error_rate(train_x, y, train_size, svm);
        error_rate_single = error_rate(valid_x, valid__y, valid_size, svm); // by validation
        if (error_rate_single < best_error){
            best_error = error_rate_single;
            for(i=0;i<img_size;i++)
                best_w[i] = w[i];
        }*/
        


        // float gpu_error_rate(float* w,float* img_t, int* y, float b, int ts , int c);
        float d_error_rate_single = gpu_error_rate(d_w[c], d_valid_x_trans, d_valid__y[c], b, valid_size,c);
        
        if (best_error_[c] > d_error_rate_single){
            cudaMemcpy(d_best_w[c], d_w[c], sizeof(float)*img_size, cudaMemcpyDeviceToDevice);
            best_error_[c] = d_error_rate_single;
        }

        /*
        // check error_rate
        if (abs(d_error_rate_single - error_rate_single) > 2e-3){
            printf("different error rate !  iter=%d\n", iter); 
            printf("d= %f     h= %f\n",d_error_rate_single, error_rate_single);
            exit(1);
        } else{
        }*/
        

        // check_difference(d_best_w[c], best_w, 1, img_size);

        printf("error rate for svm = %f\n", error_rate_single);
        float d_curr_loss = gpu_svm_loss(c);
        float curr_loss = d_curr_loss;

        /*
        float curr_loss = svm_loss();
        if (abs(d_curr_loss-curr_loss) > 1e-3){
            printf("different loss !  iter=%d\n", iter);
            printf("d= %f     h= %f\n",d_curr_loss,curr_loss);
            exit(1);
        } */


        printf("loss(to max) = %f      by gpu = %f      deta = %f\n", curr_loss, d_curr_loss, curr_loss - prev);
        if (isnan(curr_loss) || iter > 0 && prev > curr_loss + 0.00001f) {
            // throw exception("algorithm wrong !!!\n");
            printf("very wrong !!!  %d \n", iter);
            // exit(1);
        }
        prev = curr_loss;
        printf("=====================\n");
        //myfile << to_string(error_rate_single) << endl;
        error_rate_acc += error_rate_single / (10*train_size);
        

    }
    printf("error_rate_acc: %lf\n", error_rate_acc);
    
    b_gpu[c] = b;
}





// update w
void w_update(float img[][img_size]) {
    int i, j;
    // w
    for (i = 0; i < img_size; i++)
        w[i] = 0.0;
    for (i = 0; i < train_size; i++)
        for (j = 0; j < img_size; j++)
            w[j] += a[i] * train__y[i] * img[i][j]; // y shared,  a from global coalesce to share, img from global coalesce
    /*
    for (i = 0; i < img_size; i++)
        w[i] /= train_size;
     */
}


// using__binary
int svm(float img[]) { // similar to w_update with transpose,  sum up error by kernel_sum
    float res = b;
    int i;
    for (i = 0; i < img_size; i++)
        res += img[i] * w[i];

    // printf("res = %f\n",res);...directly check from here
    if (res > 0.0) // not compre with 0
        return 1;
    else
        return -1;

    // return (int)1;
}

// comprehensive...find maximum as label
int svm_general(float img[]) { // similar to w_update with transpose,  sum up error by kernel_sum
    // ge
    int c, i, best_c;
    float res, res_best;
    best_c = -1;
    for(c=0;c<class_num;c++){
        res = b_s[c];
        for(i=0;i<img_size;i++)
            res += img[i] * w_s[c][i];
        if(best_c == -1){ // no value yet
            best_c = c;
            res_best = res;
        }
        if (res_best < res){
            res_best = res;
            best_c = c;
        }
    }

    return best_c;
}










// for gpu
// allocate memory and move the data to cuda
void d_overall_init(){
    int i;
    // move and allocate the memory of general data 
    // test_x, train_x, valid_x
    // test_t, train_t, valid_t
    // how about out ???...out until the end ???

    // allocate
    cudaMalloc((void **)& d_test_x, sizeof(float)*test_size*img_size);
    cudaMalloc((void **)& d_train_x, sizeof(float)*train_size*img_size);
    cudaMalloc((void **)& d_valid_x, sizeof(float)*valid_size*img_size);
    cudaMalloc((void **)& d_test_t, sizeof(int)*test_size);
    cudaMalloc((void **)& d_train_t, sizeof(int)*train_size);
    cudaMalloc((void **)& d_valid_t, sizeof(int)*valid_size);
    
    

    // move data
    cudaMemcpy(d_test_x,test_x,sizeof(float)*test_size*img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_x,train_x,sizeof(float)*train_size*img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_valid_x,valid_x,sizeof(float)*valid_size*img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_t,test_t,sizeof(int)*test_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_t,train_t,sizeof(int)*train_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_valid_t,valid_t,sizeof(int)*valid_size, cudaMemcpyHostToDevice);

    // transpose d_train_x
    // printf("now transpose...\n");
    cudaMalloc((void **)& d_train_x_trans, sizeof(float)*train_size*img_size);
    cudaMalloc((void **)& d_valid_x_trans, sizeof(float)*valid_size*img_size);
    cudaMalloc((void **)& d_test_x_trans, sizeof(float)*test_size*img_size);
    d_transpose<<<dim3((img_size+BDIMX-1)/BDIMX,(train_size+BDIMY-1)/BDIMY),dim3(BDIMX,BDIMY)>>>(d_train_x_trans,d_train_x, img_size, train_size);
    d_transpose<<<dim3((img_size+BDIMX-1)/BDIMX,(valid_size+BDIMY-1)/BDIMY),dim3(BDIMX,BDIMY)>>>(d_valid_x_trans,d_valid_x, img_size, valid_size);
    d_transpose<<<dim3((img_size+BDIMX-1)/BDIMX,(test_size+BDIMY-1)/BDIMY),dim3(BDIMX,BDIMY)>>>(d_test_x_trans,d_test_x, img_size, test_size);
    

    // for temp use
    cudaMalloc((void **)& d_y_est_forc, sizeof(int)*class_num*max_size);
    



    // kernel...compute
    cudaMalloc((void **)& d_kernel, sizeof(float)*train_size*train_size);
    
    for(i=0;i<train_size;i++)
        d_d1_d2<<<(train_size+b_len-1)/b_len,b_len>>>(&d_train_x[i*img_size],d_train_x_trans,&d_kernel[i*train_size],0.0f,img_size,train_size);
    
}


void check_difference(float *d, float *h, int is,int js){
    int i,j;
    float d_test[is*js];
    /* 
    for(i=0;i<is;i++)
        for(j=0;j<js;j++)
            d_test[i*js+j] = 0.0;
   */
    printf("copy num=%d\n",sizeof(float)*is*js);
    cudaMemcpy(d_test, d, sizeof(float)*is*js, cudaMemcpyDeviceToHost);
    /*
    int work = 0;
    for(i=0;i<is;i++)
        for(j=0;j<js;j++)
            if(abs(d_test[i*js+j]) > 1e-3)
                work =1;
    if (work==0){
        printf("error ! not work !\n");
        exit(1);
    }*/

    for(i=0;i<is;i++)
        for(j=0;j<js;j++){
            if(abs(d_test[i*js+j]-h[i*js+j]) > 1e-3)
            {
                printf("different !\n");
                printf("i=%d  j=%d   idx=%d  d=%f    h=%f   diff=%f\n",i,j,i*js+j,d_test[i*js+j],h[i*js+j], abs(d_test[i*js+j]-h[i*js+j]));
                exit(1);
            }
        }

    printf("no difference\n");
    exit(1);
}



void d_svm_init(float myc, int c){
    // 
    C = myc;
    best_error_[c] = 1.0;


    // allocate for test__y, train__y, valid__y
    cudaMalloc((void **)& d_test__y[c], sizeof(int)*test_size);
    cudaMalloc((void **)& d_train__y[c], sizeof(int)*train_size);
    cudaMalloc((void **)& d_valid__y[c], sizeof(int)*valid_size);

    // t_y...convert ... =c -> 1....-1
    // __global__ void d_t_y(int* t, int* y, const int c, const int ts)
    d_t_y<<<(train_size+b_len-1)/b_len,b_len>>>(d_train_t,d_train__y[c],c,train_size);
    d_t_y<<<(valid_size+b_len-1)/b_len,b_len>>>(d_valid_t,d_valid__y[c],c,valid_size);
    d_t_y<<<(test_size+b_len-1)/b_len,b_len>>>(d_test_t,d_test__y[c],c,test_size);


    // b
    b_gpu[c] = 0.0;

    cudaMalloc((void **)& d_a[c], sizeof(float)*train_size);
        // if not initialized, then can't memcpy any data, just not work, hard to find

        // w...initialize 0.0
    cudaMalloc((void **)& d_w[c], sizeof(float)*img_size);
        // best_w
    cudaMalloc((void **)& d_best_w[c], sizeof(float)*img_size);


        // copy data
    cudaMemcpy(d_a[c],a,sizeof(float)*train_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w[c],w,sizeof(float)*img_size, cudaMemcpyHostToDevice);

        // for temp use
    cudaMalloc((void **)& d_ayk[c], sizeof(float)*train_size);
    cudaMalloc((void **)& d_ay[c], sizeof(float)*train_size);
    cudaMalloc((void **)& d_y_est[c], sizeof(float)*max_size);
}

void d_free(){
    cudaFree(d_test_x);
    cudaFree(d_train_x);
    cudaFree(d_train_x_trans);
    cudaFree(d_valid_x);
    cudaFree(d_kernel);

    cudaFree(d_test_t);
    cudaFree(d_train_t);
    cudaFree(d_valid_t);

    // tempuse
    cudaFree(d_y_est_forc);

    int i;
    for(i=0;i<class_num; i++){
        cudaFree(d_test__y[i]);
        cudaFree(d_train__y[i]);
        cudaFree(d_valid__y[i]);

        cudaFree(d_a[i]);
        // cudaFree(d_b[i]);
        cudaFree(d_w[i]);
        cudaFree(d_best_w[i]);

        // temp use
        cudaFree(d_ayk[i]);
        cudaFree(d_ay[i]);
        cudaFree(d_y_est[i]);
    }
}



// gpu_d1_3_multi(d_a[c], d_train__y[c], &d_kernel[j1*train_size], train_size,c)

// &d_a[c*train_size], &d_train__y[c*train_size], &d_kernel[c*train_size]
float gpu_d1_3_multi(float* a, int* y, float* kernel, int ts, int c){
    d_d1_3_multi<<<(ts+b_len-1)/b_len, b_len>>>(a,y,kernel,d_ayk[c],ts);
    
    return gpu_d1_sum(d_ayk[c], ts);
    /*
    float * d_sum, sum;
    cudaMalloc((void **)& d_sum, sizeof(float));
    d_d1_sum<<<1, b_len>>>(d_ayk[c], d_sum, ts);
    cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);

    return sum;*/
}



// svm loss checking
float x__svm_loss() { // no need, just for test
    int i, j;
    float res = 0;
    // kernel
    for (i = 0; i < train_size; i++)
        for (j = 0; j < train_size; j++)
            res += a[i] * a[j] * train__y[i] * train__y[j] * kernel[i][j];
    res *= -0.5;
    // a
    for (i = 0; i < train_size; i++)
        res += a[i];
    return (res);
}

float gpu_svm_loss(int c){
    d_d1_2_multi<<<(train_size+b_len-1)/b_len,b_len>>>(d_a[c],d_train__y[c],d_ay[c],train_size);
    d_d1_d2<<<(train_size+b_len-1)/b_len,b_len>>>(d_ay[c], d_kernel, d_ayk[c], 0.0, train_size, train_size);
    d_d1_2_multi<<<(train_size+b_len-1)/b_len,b_len>>>(d_ay[c],d_ayk[c],d_ayk[c],train_size);
    
    float res = gpu_d1_sum(d_ayk[c],train_size)*(-0.5);
    res += gpu_d1_sum_nonvolitile(d_a[c],train_size);

    return res;
}



// several cuda kernel for reuse

__global__ void d_d1_2_multi(float* a, int* y, float* res, const int ts){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ts){
        res[idx] = a[idx] * y[idx]; 
    } 
}

__global__ void d_d1_2_multi(float* a, float* y, float* res, const int ts){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ts){
        res[idx] = a[idx] * y[idx]; 
    } 
}

__global__ void d_d1_3_multi(float* a, int* y, float* kernel, float* res, const int ts){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ts){
        res[idx] = a[idx] * y[idx] * kernel[idx];
    }
}

__global__ void d_transpose(float *out, float *in, int nx, int ny){
    // __shared__ float tile[BDIMY][BDIMX];

    unsigned int ix,iy;//,ti,to;

    ix = blockIdx.x *blockDim.x + threadIdx.x;
    iy = blockIdx.y *blockDim.y + threadIdx.y;
    if (ix<nx && iy <ny){  // ix write as iy...error
        out[ix*ny+iy] = in[iy*nx+ix];
    }

    /*
    if(iy==0 && ix==152){
        // printf("now = %d   %f  previous = %d   %f");
        printf("now = %d   %f  \n previous = %d   %f\n", ix*ny+iy,out[ix*ny+iy], iy*nx+ix , in[iy*nx+ix]);
        printf("ix= %d   iy= %d   nx= %d   ny= %d\n",ix,iy,nx,ny);
        printf("compare result %d %d\n",ix<nx, ix <ny);
    }
    */
    

    /*
    something wrong with this code
    // linear global memory index for original matrix
    ti = iy*nx + ix;
    // thread index in transposed block
    unsigned int bidx, irow, icol;
    bidx = threadIdx.y*blockDim.x + threadIdx.x;
    irow = bidx/blockDim.y;
    icol = bidx%blockDim.y;
    // coordinate in transposed matrix
    ix = blockIdx.y * blockDim.y + icol;
    iy = blockIdx.x * blockDim.x + irow;
    // linear global memory index for transposed matrix
    to = iy*ny + ix;
    // transpose with boundary test
    if (ix < nx && iy < ny) {
        // load data from global memory to shared memory
        tile[threadIdx.y][threadIdx.x] = in[ti];
        // thread synchronization
        //printf("prev i=%d  j=%d   ")
        __syncthreads();
        // store data to global memory from shared memory
        out[to] = tile[icol][irow];
    }
    */
}

__global__ void d_d1_d2(float *w, float *img_t, float *y_est,const float b, const int ws, const int ts) {
    // actually here is img_t
    int idx = blockIdx.x*blockDim.x +threadIdx.x;

    // use coalesece access the w, img...much more quickly, no need for shared memory at all
    int i=0;
    float res=0;
    if (idx < ts){
        for(i=0;i<ws;i++)
            res += w[i]*img_t[i*ts+idx];
        y_est[idx] = res+b;
        // printf("idx= %d     res= %f     ts=%d\n",idx,y_est[idx],ts);
    }

}

__global__ void d_d1_d1_y_comp(int *y, float *y_est, const int ts){
    // compare y with y_est
    // note y_est is not in -1, 1
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int y_;
    float y_est_; // convert type....no warning but cause error !!! pervious int y_est_;
    if(idx < ts){
        y_ = y[idx];
        y_est_ =y_est[idx];
        float res = 0.0;
        if (y_est_<0.0f && y_==-1)
            res = 1.0;
        else if (y_est_>=0.0f && y_==1)
            res = 1.0;
        y_est[idx] = res;
        // printf("idx= %d      %f      %d      score= %f \n",idx, y_est_, y_,y_est[idx]);
    }
}

float gpu_d1_sum_nonvolitile(float* x, const int ts){
    // in order to keep origin data unchanged
    float *x_copy;
    cudaMalloc((void **)& x_copy, sizeof(float)*ts);
    cudaMemcpy(x_copy, x, sizeof(float)*ts, cudaMemcpyDeviceToDevice);
    float res = gpu_d1_sum(x_copy, ts);
    cudaFree(x_copy);

    return res;
}

float gpu_d1_sum(float* x, const int ts){
    float * d_sum, sum;
    cudaMalloc((void **)& d_sum, sizeof(float));
    d_d1_sum<<<1, b_len>>>(x, d_sum, ts);
    cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);
    return sum;
}


__global__ void deleted_d_d1_sum(float* x, float* sum, const int ts){
    // x won't be reused, just temp, thus we can just use it !!!
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // int s = ts/2;
    int s = (ts+1)/2;
    int prev_s = ts;
    // printf("executed   idx=%d   s=%d   now=%f\n", idx,s,x[idx]);
    for (;prev_s>1; prev_s=s, s = (s+1)/2){  // not >1, but >0 !!! s=1,, x[1] += x[1+1]
       // printf("s=%d\n",s);
       if (idx<s && idx+s < prev_s){
              x[idx] += x[idx+s];
              // printf("idx=%d   s=%d   now=%f\n", idx,s,x[idx]);
       }
        __syncthreads();
    }

    if(idx ==0){
        *sum = x[0];
    }
}


__global__ void d_d1_sum(float* x, float* sum, const int ts){
    // since __syncthreads() only synchronize within a block, I'd better only use 1 block
    int idx = threadIdx.x;
    int bs = blockDim.x;
    int i;
    int is=(ts+bs-1)/bs;
    float temp=0.0;
    for (i=0;i<is;i++)
       if (idx + i*bs < ts){
              temp += x[idx + i*bs];
       }
    __syncthreads();
    x[idx] = temp;
    __syncthreads();


    // int s = ts/2;
    int s = (bs+1)/2;
    int prev_s = bs;
    // printf("executed   idx=%d   s=%d   now=%f\n", idx,s,x[idx]);
    for (;prev_s>1; prev_s=s, s = (s+1)/2){  // not >1, but >0 !!! s=1,, x[1] += x[1+1]
       // printf("s=%d\n",s);
       if (idx<s && idx+s < prev_s){
              x[idx] += x[idx+s];
              // printf("idx=%d   s=%d   now=%f\n", idx,s,x[idx]);
       }
       __syncthreads();
    }

    if(idx ==0){
        *sum = x[0];
    }
}

__global__ void d_t_y(int* t, int* y, const int c, const int ts){
    // convert t(0-9) to y(-1,1)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < ts){
        if(t[idx] == c)
            y[idx] = 1;
        else
            y[idx] = -1;
    }
}




void check_difference(float a, float b){
    if(abs(a-b)>1e-3){
        printf("different result !!!!!-------------------- \n");
        printf("%f    %f\n",a,b);
        exit(1);
    }
}


int main() {
    read_mnist("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", test_x, test_t,test_size);
    read_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte", train_x, train_t, train_size);
    printf("%d    %d   %d   %d\n", magic_number, number_of_images, n_rows, n_cols);
    print_img(train_x[0]);
    printf("%d\n", train_t[0]);
    overall_init();
    d_overall_init();

    // must first initalize
    start_t = clock();
    printf("initalize .....\n");
    int target;
    float error_test[10][2];
    /**/
    for (target = 0; target <10;target++){ 
        svm_init(0.2, train_x, train_t, target);
        printf("train .....%d\n", target);
        svm_SMO_new(train_x, train__y);
        // record b_s
        // printf("b=%f\n",b);
        b_s[target] = b;
        // use best_w
        arr_cp(w,best_w,img_size);
        // for overall comprehensive
        arr_cp(w_s[target], best_w, img_size);

        error_test[target][0]=error_rate(test_x, test__y, test_size, svm);
        b = 0;
        error_test[target][1]=error_rate(test_x, test__y, test_size, svm); // without b

        printf("error rate for svm = %f\n", error_test[target][0]);
    }
    end_t = clock();
    cpu_time = ((double) (end_t - start_t)) / CLOCKS_PER_SEC;

    start_t = clock();
    float d_error_test[10][2];
    for (target = 0; target <10;target++){ 
        printf("cuda initialization for %d.....\n", target);
        svm_init(0.2, train_x, train_t, target);
        d_svm_init(0.2, target); 
        printf("cuda train .....%d\n", target);
        svm_SMO_cuda(target, train_x, train__y);

        // use best w
        cudaMemcpy(d_w[target], d_best_w[target], sizeof(float)*img_size, cudaMemcpyDeviceToDevice);

        

        d_error_test[target][0]=gpu_error_rate(d_w[target], d_test_x_trans, d_test__y[target], b_gpu[target], test_size,target);
        d_error_test[target][1]=gpu_error_rate(d_w[target], d_test_x_trans, d_test__y[target], 0.0, test_size,target);

        // printf("error rate for svm = %f\n", error_test[target][0]);
    }
    end_t = clock();
    gpu_time = ((double) (end_t - start_t)) / CLOCKS_PER_SEC;
    printf("time used:    cpu = %f      gpu= %f\n", cpu_time, gpu_time);

    /**/
    int i=0;
    for(i=0;i<class_num;i++)
        // printf("i=%d    b_s=%f   b_gpu=%f\n",i,b_s[i],b_gpu[i]);
        check_difference(b_s[i],b_gpu[i]);
    

    for (target = 0; target <10;target++){
        printf("# %d   error rate for svm = %f      without b = %f\n", target,error_test[target][0],error_test[target][1]);
        printf("                  by cuda = %f      without b = %f\n",d_error_test[target][0],d_error_test[target][1]);
        check_difference(error_test[target][0],d_error_test[target][0]);
        check_difference(error_test[target][1],d_error_test[target][1]);
    }


    // check overall comprehensive error

    printf("comprehensive error = %f\n",error_rate_general(test_x, test_t, test_size));
    printf("             by svm = %f\n",gpu_error_rate_general(d_test_x_trans, d_test_t, test_size));
    check_difference(error_rate_general(test_x, test_t, test_size), gpu_error_rate_general(d_test_x_trans, d_test_t, test_size));

    d_free();
}




