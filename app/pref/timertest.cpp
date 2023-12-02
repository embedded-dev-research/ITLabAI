#include <omp.h>
#include <time.h>

#include <chrono>
#include <iostream>
#include <thread>

using namespace std;

inline void counter1(int i, int n, double* arr) {
  clock_t timer;
  double res;
  timer = clock();
  this_thread::sleep_for(chrono::milliseconds(n));
  res = (double)(clock() - timer) / CLOCKS_PER_SEC;
  arr[i] = res;
}

inline void counter2(int i, int n, double* arr) {
  auto chronotimer = chrono::high_resolution_clock();
  auto duration = chrono::duration<double, milli>();
  auto start = chronotimer.now();
  this_thread::sleep_for(chrono::milliseconds(n));
  auto end = chronotimer.now();
  duration = end - start;
  arr[i] = duration.count();
}

inline void counter3(int i, int n, double* arr) {
  double duration;
  double start = omp_get_wtime();
  this_thread::sleep_for(chrono::milliseconds(n));
  double end = omp_get_wtime();
  duration = end - start;
  arr[i] = duration;
}

const int ITERS = 100;

int main() {
  int n = 2000;
  thread threads[ITERS];
  double arr[ITERS];
  char choice;
  cin >> choice;
  this_thread::sleep_for(chrono::milliseconds(3000));
  if (choice == '1') {
    for (int i = 0; i < ITERS; i++) {
      threads[i] = thread(counter1, i, n, arr);
    }
    for (int i = 0; i < ITERS; i++) {
      threads[i].join();
    }
    for (int i = 0; i < ITERS; i++) {
      cout << i + 1 << ") " << arr[i] << endl;
    }
  } else if (choice == '2') {
    for (int i = 0; i < ITERS; i++) {
      threads[i] = thread(counter2, i, n, arr);
    }
    for (int i = 0; i < ITERS; i++) {
      threads[i].join();
    }
    for (int i = 0; i < ITERS; i++) {
      cout << i + 1 << ") " << arr[i] << endl;
    }
  } else if (choice == '3') {
    #pragma omp parallel for num_threads(ITERS)
    for (int i = 0; i < ITERS; i++) {
      counter3(i, n, arr);
    }
    for (int i = 0; i < ITERS; i++) {
      cout << i + 1 << ") " << arr[i] << endl;
    }
  } else if (choice == '4') {
    for (int i = 0; i < ITERS; i++) {
      threads[i] = thread(counter3, i, n, arr);
    }
    for (int i = 0; i < ITERS; i++) {
      threads[i].join();
    }
    for (int i = 0; i < ITERS; i++) {
      cout << i + 1 << ") " << arr[i] << endl;
    }
  }
  return 0;
}
