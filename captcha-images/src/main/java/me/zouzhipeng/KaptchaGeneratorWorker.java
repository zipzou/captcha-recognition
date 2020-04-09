package me.zouzhipeng;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Properties;
import java.util.Random;

import com.google.code.kaptcha.Constants;
import com.google.code.kaptcha.Producer;
import com.google.code.kaptcha.util.Config;

import me.zouzhipeng.utils.ImageOutputUtil;

public class KaptchaGeneratorWorker implements CaptchaGenerator {
  private Random rand = new Random();
  private Producer producer;

  public KaptchaGeneratorWorker() {
    Properties prop = new Properties();
    prop.put(Constants.KAPTCHA_BORDER, true);
    prop.put(Constants.KAPTCHA_BORDER_COLOR,
        String.join(",", rand.nextInt(256) + "", rand.nextInt(256) + "", rand.nextInt(256) + ""));
    prop.put(Constants.KAPTCHA_IMAGE_WIDTH, "200");
    prop.put(Constants.KAPTCHA_IMAGE_HEIGHT, "50");
    String fontColor = String.join(",", rand.nextInt(256) + "", rand.nextInt(256) + "", rand.nextInt(256) + "");
    prop.put(Constants.KAPTCHA_TEXTPRODUCER_FONT_COLOR,
        fontColor);
    prop.put(Constants.KAPTCHA_TEXTPRODUCER_CHAR_LENGTH, "4");
    prop.put(Constants.KAPTCHA_TEXTPRODUCER_FONT_NAMES, "彩云,宋体,楷体,微软雅黑,Arial,SimHei,SimKai,SimSum");
    prop.put(Constants.KAPTCHA_NOISE_COLOR,
        fontColor);
    prop.put(Constants.KAPTCHA_TEXTPRODUCER_CHAR_STRING, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz012345679");
    Config kaptchaConfig = new Config(prop);
    producer = kaptchaConfig.getProducerImpl();
  }

  @Override
  public boolean generate(String folder) {
    String text = producer.createText();
    BufferedImage imageBuffered = producer.createImage(text);

    return ImageOutputUtil.writeToFile(imageBuffered, folder, text, "jpg");

  }

  @Override
  public boolean generate() {
    String cwd = System.getProperty("user.dir");

    return generate(new File(cwd, "kaptchas").getAbsolutePath());
  }
}