package me.zouzhipeng;

import java.awt.image.BufferedImage;
import java.util.Properties;
import java.util.Random;

import com.google.code.kaptcha.Constants;
import com.google.code.kaptcha.Producer;
import com.google.code.kaptcha.util.Config;

import org.apache.log4j.Logger;

import me.zouzhipeng.config.ConfigConstants;
import me.zouzhipeng.utils.ImageOutputUtil;

public class KaptchaGeneratorWorker implements CaptchaGenerator {
  private Random rand = new Random();
  private Producer producer;
  private String output;
  private static final Logger LOG = Logger.getLogger(KaptchaGeneratorWorker.class);

  public KaptchaGeneratorWorker(me.zouzhipeng.config.Config config) {
    Properties prop = new Properties();
    prop.put(Constants.KAPTCHA_BORDER, true);
    prop.put(Constants.KAPTCHA_BORDER_COLOR,
        String.join(",", rand.nextInt(256) + "", rand.nextInt(256) + "", rand.nextInt(256) + ""));
    prop.put(Constants.KAPTCHA_IMAGE_WIDTH, config.get(ConfigConstants.WIDTH, "200"));
    prop.put(Constants.KAPTCHA_IMAGE_HEIGHT, config.get(ConfigConstants.HEIGHT, "50"));
    String textColor = config.get(ConfigConstants.TEXT_COLOR);
    if (null == textColor) {
      textColor = String.join(",", rand.nextInt(256) + "", rand.nextInt(256) + "", rand.nextInt(256) + "");
    }
    prop.put(Constants.KAPTCHA_TEXTPRODUCER_FONT_COLOR,
        textColor);
    prop.put(Constants.KAPTCHA_TEXTPRODUCER_CHAR_LENGTH, config.get(ConfigConstants.LENGTH, "4"));
    prop.put(Constants.KAPTCHA_TEXTPRODUCER_FONT_NAMES, "彩云,宋体,楷体,微软雅黑,Arial,SimHei,SimKai,SimSum");
    if (Boolean.parseBoolean(config.get(ConfigConstants.NOISE_SAME_TEXT_COLOR, "true"))) {
      prop.put(Constants.KAPTCHA_NOISE_COLOR, textColor);
    } else {
      prop.put(Constants.KAPTCHA_NOISE_COLOR,
        String.join(",", rand.nextInt(256) + "", rand.nextInt(256) + "", rand.nextInt(256) + ""));
    }
    prop.put(Constants.KAPTCHA_TEXTPRODUCER_CHAR_STRING, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789");
    this.output = config.get(ConfigConstants.OUT_DIR);
    Config kaptchaConfig = new Config(prop);
    producer = kaptchaConfig.getProducerImpl();
  }

  @Override
  @Deprecated
  public boolean generate(String folder) {
    String text = producer.createText();
    BufferedImage imageBuffered = producer.createImage(text);
    if (LOG.isInfoEnabled()) {
      LOG.info("Generating " + text + "...");
    }
    return ImageOutputUtil.writeToFile(imageBuffered, folder, text, "jpg");

  }

  @Override
  public boolean generate() {
    String text = producer.createText();
    BufferedImage imageBuffered = producer.createImage(text);

    return ImageOutputUtil.writeToFile(imageBuffered, this.output, text, "jpg");
  }
}