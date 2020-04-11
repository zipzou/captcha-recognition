package me.zouzhipeng;

import java.awt.FontFormatException;
import java.io.IOException;

import com.wf.captcha.SpecCaptcha;
import com.wf.captcha.base.Captcha;

import org.apache.log4j.Logger;

import me.zouzhipeng.config.Config;
import me.zouzhipeng.config.ConfigConstants;
import me.zouzhipeng.utils.ImageOutputUtil;

public class EasyCaptchaGeneratorWorker implements CaptchaGenerator {

  private Config config;

  private static final Logger LOG = Logger.getLogger(EasyCaptchaGeneratorWorker.class);

  public EasyCaptchaGeneratorWorker(Config config) {
    this.config = config;
  }


  @Override
  public boolean generate(String path) {
    SpecCaptcha captcha = new SpecCaptcha(120, 80, 4);
    captcha.setCharType(Captcha.TYPE_DEFAULT);
    try {
      captcha.setFont(Captcha.FONT_3);
    } catch (IOException | FontFormatException e1) {
      e1.printStackTrace();
      return false;
    }
    String codes = captcha.text();
    return ImageOutputUtil.writeToFile(captcha, path, codes);
  }

  @Override
  public boolean generate() {
    String outputFolder = config.get(ConfigConstants.OUT_DIR);
    int width = Integer.parseInt(config.get(ConfigConstants.WIDTH, "120"));
    int height = Integer.parseInt(config.get(ConfigConstants.HEIGHT, "80"));
    int len = Integer.parseInt(config.get(ConfigConstants.LENGTH));
    SpecCaptcha captcha = new SpecCaptcha(width, height, len);
    captcha.setCharType(Captcha.TYPE_DEFAULT);
    try {
      captcha.setFont(Captcha.FONT_3);
    } catch (IOException | FontFormatException e1) {
      e1.printStackTrace();
      return false;
    }
    String codes = captcha.text();
    if (LOG.isInfoEnabled()) {
      LOG.info("Generating " + codes + "...");
    }
    return ImageOutputUtil.writeToFile(captcha, outputFolder, codes);
  }
  
}