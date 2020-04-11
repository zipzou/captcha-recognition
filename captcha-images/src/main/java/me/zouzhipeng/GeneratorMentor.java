package me.zouzhipeng;

import java.util.Properties;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.log4j.Logger;

import me.zouzhipeng.config.Config;
import me.zouzhipeng.config.ConfigConstants;

public class GeneratorMentor implements Generator {

  private static final Logger LOG = Logger.getLogger(GeneratorMentor.class);
  @Override
  public void generate(Config config) {
    int count = Integer.parseInt(config.get(ConfigConstants.COUNT));
    int kind = Integer.parseInt(config.get(ConfigConstants.KIND));

    String mode = config.get(ConfigConstants.MODE);

    if (mode.equals(ConfigConstants.EASY_CAPTCHA)) {
      startWorking(mode, count, kind, config);
      if (LOG.isInfoEnabled()) {
        LOG.info("Starting task, current mode: " + mode + ".");
      }
    } else if (mode.equals(ConfigConstants.KAPTCHA)) {
      startWorking(mode, count, kind, config);
      if (LOG.isInfoEnabled()) {
        LOG.info("Starting task, current mode: " + mode + ".");
      }
    } else {
      if (LOG.isInfoEnabled()) {
        LOG.info("WARN: " + mode + " is not be supported, skip this task.");
      }
    }

  }

  @Override
  public void generate(Properties prop) {
    Config cfg = new Config(prop);
    generate(cfg);
  }

  /**
   * 
   * @param mode
   * @param count
   * @param kind
   * @param config
   * @throws NullPointerException 
   */
  protected void startWorking(String mode, int count, int kind, final Config config) {
    int size = Integer.parseInt(config.get(ConfigConstants.POOL_SIZE, "20"));
    ExecutorService pool = Executors.newFixedThreadPool(size);
    CaptchaGenerator generator = null;
    for (int i = 0; i < count; i++) {
      CaptchaTaskRunner runner = new CaptchaTaskRunner();
      if (i % (count / kind) == 0) {
        if (mode.equals(ConfigConstants.KAPTCHA)) {
          generator = new KaptchaGeneratorWorker(config);
        } else if (mode.equals(ConfigConstants.EASY_CAPTCHA)) {
          generator = new EasyCaptchaGeneratorWorker(config);
        } else {
          NullPointerException nullGeneratorEx = new NullPointerException("Unsupported mode made generator null.");
          if (LOG.isTraceEnabled()) {
            LOG.trace(nullGeneratorEx);
          }
          throw nullGeneratorEx;
        }
      }
      runner.setGenerator(generator);
      pool.submit(runner);
    }
    pool.shutdown();
  }
}