package me.zouzhipeng.utils;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import javax.imageio.ImageIO;

import com.wf.captcha.base.Captcha;

public class ImageOutputUtil {
  /**
   * To write buffered image to file.
   * @param image the image to save
   * @param folder the folder to save images
   * @param name the name of the image
   * @param extension the extension of the image, i.e. jpg
   * @return successful or failed after output
   */
  public static boolean writeToFile(BufferedImage image, String folder, String name, String extension) {
    File imageFolder = new File(folder);
    if (!imageFolder.exists()) {
      imageFolder.mkdirs();
    }
    File imageFile = new File(imageFolder, name + "." + extension);
    if (imageFile.exists()) {
      return false;
    }
    FileOutputStream imageOutput = null;
    try {
      imageOutput = new FileOutputStream(imageFile);
      ImageIO.write(image, extension, imageOutput);
      return true;
    } catch (FileNotFoundException e) {
      e.printStackTrace();
      return false;
    } catch (IOException e) {
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

  /**
   * To write captcha image to an image file.
   * @param captcha the captcha to write
   * @param folder the folder to save captchas
   * @param name the name of the captcha
   * @return successful or failed after output
   */
  public static boolean writeToFile(Captcha captcha, String folder, String name) {
    File imageFolder = new File(folder);
    if (!imageFolder.exists()) {
      imageFolder.mkdirs();
    }
    File imageFile = new File(imageFolder, name + ".jpg");
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
}