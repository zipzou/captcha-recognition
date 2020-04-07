package me.zouzhipeng;

import java.awt.FontFormatException;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import com.wf.captcha.SpecCaptcha;
import com.wf.captcha.base.Captcha;

public class CaptchaGeneratorWorker implements CaptchaGenerator {

  @Override
  public boolean generate(String path) {
    SpecCaptcha captcha = new SpecCaptcha(120, 80, 4);
    captcha.setCharType(Captcha.TYPE_NUM_AND_UPPER);
    try {
      captcha.setFont(Captcha.FONT_3);
    } catch (IOException | FontFormatException e1) {
      e1.printStackTrace();
      return false;
    }
    String codes = captcha.text();
    File imageFolder = new File(path);
    if (!imageFolder.exists()) {
      imageFolder.mkdirs();
    }
    File imageFile = new File(imageFolder, codes + ".jpg");
    if (imageFile.exists()) {
      return false;
    }
    FileOutputStream imageOutput = null;
    try {
      imageOutput = new FileOutputStream(imageFile);
      captcha.out(imageOutput);
      return true;
    } catch (FileNotFoundException e) {
      e.printStackTrace();
      return false;
    } finally {
      if (null != imageOutput) {
        try {
          imageOutput.close();
        } catch (IOException e) {
          e.printStackTrace();
          return false;
        }
      }
    }
  }

  @Override
  public boolean generate() {
    String currentWorkspace = System.getProperty("user.dir");
    return generate(currentWorkspace);
  }
  
}