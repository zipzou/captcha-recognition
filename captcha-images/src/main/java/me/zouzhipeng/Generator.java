package me.zouzhipeng;

import java.util.Properties;

import me.zouzhipeng.config.Config;

public interface Generator {
  /**
   * 
   * @param config
   */
  public void generate(Config config);
  /**
   * 
   * @param prop
   */
  public void generate(Properties prop);
}