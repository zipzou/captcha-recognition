package me.zouzhipeng;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Hello world!
 *
 */
public class App {
  public static void main(String[] args) {
    // generateCaptcha("/Users/frank/Downloads/captchas", 50000);
    generateKaptcha("/Users/frank/Downloads/kaptchas-test", 10, 10/2);
  }
  public static void generateKaptcha(String folder, int count, int kind) {
    ExecutorService pool = Executors.newFixedThreadPool(20);
    CaptchaGenerator generator = null;
    for (int i = 0; i < count; i++) {
      KaptchaRunner runner = new KaptchaRunner();
      if ((i + 1) % kind == 0 && (i > 0)) {
        generator = new KaptchaGeneratorWorker();
      }
      runner.setGenerator(generator);
      runner.setOutput(folder);
      pool.submit(runner);
    }
    pool.shutdown();
  }

  public static void generateCaptcha(final String folder, int count) {
    ExecutorService pool = Executors.newFixedThreadPool(20);
    for (int i = 0; i < count; i++) {
      Runnable runner = new Runnable() {
        @Override
        public void run() {
          CaptchaGenerator generator = new EasyCaptchaGeneratorWorker();
          boolean success = generator.generate(folder);
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
