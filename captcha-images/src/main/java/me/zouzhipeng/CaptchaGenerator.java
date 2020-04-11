package me.zouzhipeng;

public interface CaptchaGenerator {
  
  /**
   * Generator captcha images automaticlly to destination.
   * @param folder the folder path to output
   * @return Successful or failed after created.
   */
  @Deprecated
  public boolean generate(String folder);

  /**
   * Generator captcha images automaticlly to current workspace.
   * @return Successful or failed after created.
   */
  public boolean generate();
}