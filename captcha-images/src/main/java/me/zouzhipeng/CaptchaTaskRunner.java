package me.zouzhipeng;

import org.apache.log4j.Logger;

public class CaptchaTaskRunner implements Runnable {

  private static final Logger LOG = Logger.getLogger(CaptchaTaskRunner.class);

  private CaptchaGenerator generator;

  @Override
  public void run() {
    boolean success = generator.generate();
    if (success) {
      if (LOG.isInfoEnabled()) {
        LOG.info("Complete!");
      }
    } else {
      if (LOG.isInfoEnabled()) {
        LOG.info("Failed!");
      }
    }
  }

  /**
   * @return CaptchaGenerator return the generator
   */
  public CaptchaGenerator getGenerator() {
    return generator;
  }

  /**
   * @param generator the generator to set
   */
  public void setGenerator(CaptchaGenerator generator) {
    this.generator = generator;
  }
}