package me.zouzhipeng;

public class KaptchaRunner implements Runnable {

  private CaptchaGenerator generator;

  private String output;

  @Override
  public void run() {
    boolean success = generator.generate(output);
    if (success) {
      System.out.println("Complete!");
    } else {
      System.out.println("Failed");
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

  /**
   * @return String return the output
   */
  public String getOutput() {
    return output;
  }

  /**
   * @param output the output to set
   */
  public void setOutput(String output) {
    this.output = output;
  }

}