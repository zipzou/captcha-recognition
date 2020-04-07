package me.zouzhipeng;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Hello world!
 *
 */
public class App {
  public static void main(String[] args) {
    int sizes = 50000;
    ExecutorService pool = Executors.newFixedThreadPool(20);
    for (int i = 0; i < sizes; i++) {
      Runnable runner = new Runnable() {
        @Override
        public void run() {
          CaptchaGenerator generator = new CaptchaGeneratorWorker();
          boolean success = generator.generate("/Users/frank/Downloads/captchas");
          if (success) {
            System.out.println("Complete!");
          } else {
            System.out.println("Failed");
          }
        }
      };
      pool.submit(runner);
    }
    pool.shutdown();
  }
}
