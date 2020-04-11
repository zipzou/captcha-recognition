package me.zouzhipeng.config;

import java.io.File;
import java.util.Properties;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.log4j.Logger;

public class Config {
  private static final Logger LOG = Logger.getLogger(Config.class);
  private Properties prop;

  public Config() {
    this.prop = new Properties();
    this.prop.setProperty(ConfigConstants.MODE, ConfigConstants.EASY_CAPTCHA);
    this.prop.setProperty(ConfigConstants.COUNT, "50000");
    this.prop.setProperty(ConfigConstants.KIND, "1");
    this.prop.setProperty(ConfigConstants.LENGTH, "4");
    String cwd = System.getProperty("user.dir");
    File outputDir = new File(cwd, "captchas");
    this.prop.setProperty(ConfigConstants.OUT_DIR, outputDir.getAbsolutePath());
  }

  public Config(Properties prop) {
    this();
    Set<Entry<Object, Object>> entries = this.prop.entrySet();
    for (Entry<Object, Object> entry : entries) {
      this.prop.put(entry.getKey(), entry.getValue());
    }
    this.prop = prop;
  }

  public String get(String key) {
    if (key.contains(key)) {
      return prop.getProperty(key);
    }
    if (LOG.isInfoEnabled()) {
      LOG.info(key + " not found!");
    }
    return null;
  }

  public String get(String key, String defaultValue) {
    String value = get(key);
    if (null == value) {
      return defaultValue;
    } else {
      return value;
    }
  }

  public void set(String key, String value) {
    this.prop.setProperty(key, value);
  }
}