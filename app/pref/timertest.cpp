#include <omp.h>
#include <time.h>

#include <chrono>
#include <iostream>
#include <thread>

using namespace std;

inline void counter1(int i, int n) {
  clock_t timer;
  timer = clock();
  this_thread::sleep_for(chrono::milliseconds(n));
  timer = clock() - timer;
  cout << timer << endl;
}

inline void counter2(int i, int n) {
  auto chronotimer = chrono::high_resolution_clock();
  auto duration = chrono::duration<double, milli>();
  auto start = chronotimer.now();
  this_thread::sleep_for(chrono::milliseconds(n));
  auto end = chronotimer.now();
  duration = end - start;
  cout << duration.count() << endl;
}

inline void counter3(int i, int n) {
  double duration;
  double start = omp_get_wtime();
  this_thread::sleep_for(chrono::milliseconds(n));
  double end = omp_get_wtime();
  duration = end - start;
  cout << duration * 1000.0 << endl;
}

const int ITERS = 100;

int main() {
  int n = 2000;
  thread threads[100];
  char choice;
  cin >> choice;
  this_thread::sleep_for(chrono::milliseconds(3000));
  if (choice == '1') {
    for (int i = 0; i < ITERS; i++) {
      threads[i] = thread(counter1, i, n);
    }
    for (int i = 0; i < ITERS; i++) {
      threads[i].join();
    }
  } else if (choice == '2') {
    for (int i = 0; i < ITERS; i++) {
      threads[i] = thread(counter2, i, n);
    }
    for (int i = 0; i < ITERS; i++) {
      threads[i].join();
    }
  } else if (choice == '3') {
    for (int i = 0; i < ITERS; i++) {
      threads[i] = thread(counter2, i, n);
    }
    for (int i = 0; i < ITERS; i++) {
      threads[i].join();
    }
  }
  return 0;
}
