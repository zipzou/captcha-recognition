package me.zouzhipeng;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import me.zouzhipeng.config.Config;
import me.zouzhipeng.config.ConfigConstants;

/**
 * 
 *
 */
public class App {
  public static void main(String[] args) {
    // generateCaptcha("/Users/frank/Downloads/captchas", 50000);
    // generateKaptcha("/Users/frank/Downloads/kaptchas-test", 10, 10/2);
    Options options = defineOptions();

    // String[] testArgs = {"-m", "kaptcha", "-t", "10"};
    Config conf = parseOptions(options, args);

    if (null == conf) {
      return;
    }

    Generator generator = new GeneratorMentor();
    generator.generate(conf);
  }

  public static Options defineOptions() {
    Option modeOption = Option.builder("m").argName("mode").desc(String
        .format("The mode of captcha, could be [%s] or [%s]", ConfigConstants.EASY_CAPTCHA, ConfigConstants.KAPTCHA))
        .hasArg().required(false).build();
    Option helpOption = Option.builder("h").argName("help").desc("Show help").required(false).hasArg(false).build();
    Option outputDirOption = Option.builder("o").argName("output_dir").hasArg()
        .desc("The output directory of captchas, string").required(false).build();

    Option widthOption = Option.builder("w").hasArg().argName("width")
        .desc("The width of the capthcas, integer, default 120").required(false).build();
    Option heightOption = Option.builder("v").hasArg().argName("height")
        .desc("The height of the capthcas, integer, default 80").required(false).build();
    Option sizeOption = Option.builder("s").argName("size")
        .desc("The size of the capthcas, should be: <width>,<height>").hasArg().numberOfArgs(2).valueSeparator(',')
        .required(false).build();
    Option textColorOption = Option.builder("c").argName("text_color")
        .desc("The text color of the captcha, if not specified, use randomly, should be <R>,<G>,<B>").hasArg()
        .required(false).build();
    Option noiseColorOption = Option.builder("n").argName("noise_color")
        .desc("The noise color of the captcha, if not specified, use randomly, should be <R>,<G>,<B>").hasArg()
        .required(false).build();
    Option lengthOption = Option.builder("l").argName("length")
        .desc("The length of characters in the captchas, integer, default 4").hasArg().required(false).build();
    Option countOption = Option.builder("t").argName("count").desc("The count to produce, integer, default 5,000")
        .hasArg().required(false).build();
    Option kindOption = Option.builder("k").argName("kinds").desc("The kinds of captchas, integer, default 1").hasArg()
        .required(false).build();
    Option poolOption = Option.builder("p").argName("pool_size").desc("Thread pool size, integer, default 20").hasArg()
        .required(false).build();
    Option sameColorOption = Option.builder("e").argName("noise_same_text")
        .desc("If the noise color is the same with the text color, should be true or false, default false.").hasArg()
        .required(false).build();

    Options options = new Options();
    options.addOption(modeOption);
    options.addOption(helpOption);
    options.addOption(outputDirOption);
    options.addOption(widthOption);
    options.addOption(heightOption);
    options.addOption(sizeOption);
    options.addOption(textColorOption);
    options.addOption(noiseColorOption);
    options.addOption(lengthOption);
    options.addOption(countOption);
    options.addOption(kindOption);
    options.addOption(poolOption);
    options.addOption(sameColorOption);

    return options;
  }

  public static Config parseOptions(Options options, String[] args) {
    HelpFormatter helpFormatter = new HelpFormatter();
    CommandLine cmdLine = null;
    CommandLineParser cmdParser = new DefaultParser();
    try {
      cmdLine = cmdParser.parse(options, args);
      if (null != cmdLine) {
        if (0 >= cmdLine.getOptions().length) {
          return new Config();
        }
        Config config = new Config();
        if (cmdLine.hasOption('h')) {
          helpFormatter.printHelp("java -jar [jarfile].jar", options);
          return null;
        } else {
          if (cmdLine.hasOption('m')) {
            String modeValue = cmdLine.getOptionValue('m');
            if (modeValue.equals(ConfigConstants.KAPTCHA) || modeValue.equals(ConfigConstants.EASY_CAPTCHA)) {
              config.set(ConfigConstants.MODE, modeValue);
            } else {
              throw new ParseException(String.format("Only %s and %s modes are supported!", ConfigConstants.EASY_CAPTCHA,
                  ConfigConstants.KAPTCHA));
            }
          } 
          if (cmdLine.hasOption('o')) {
            String outputDirValue = cmdLine.getOptionValue('o');
            config.set(ConfigConstants.OUT_DIR, outputDirValue);
          }
          if (cmdLine.hasOption('w')) {
            String widthValue = cmdLine.getOptionValue('w');
            config.set(ConfigConstants.WIDTH, widthValue);
          } 
          if (cmdLine.hasOption('v')) {
            String heightValue = cmdLine.getOptionValue('v');
            config.set(ConfigConstants.HEIGHT, heightValue);
          } 
          if (cmdLine.hasOption('s')) {
            String[] sizeValue = cmdLine.getOptionValues('s');
            config.set(ConfigConstants.WIDTH, sizeValue[0]);
            config.set(ConfigConstants.HEIGHT, sizeValue[1]);
          } 
          if (cmdLine.hasOption('c')) {
            String textColorValue = cmdLine.getOptionValue('c');
            config.set(ConfigConstants.TEXT_COLOR, textColorValue);
          } 
          if (cmdLine.hasOption('n')) {
            String noiseColor = cmdLine.getOptionValue('n');
            config.set(ConfigConstants.NOISE_COLOR, noiseColor);
          } 
          if (cmdLine.hasOption('l')) {
            String lengthValue = cmdLine.getOptionValue('l');
            config.set(ConfigConstants.LENGTH, lengthValue);
          } 
          if (cmdLine.hasOption('t')) {
            String countValue = cmdLine.getOptionValue('t');
            config.set(ConfigConstants.COUNT, countValue);
          } 
          if (cmdLine.hasOption('k')) {
            String kindCountValue = cmdLine.getOptionValue('k');
            config.set(ConfigConstants.KIND, kindCountValue);
          } 
          if (cmdLine.hasOption('p')) {
            String poolSize = cmdLine.getOptionValue('k');
            config.set(ConfigConstants.POOL_SIZE, poolSize);
          } 
          if (cmdLine.hasOption('e')) {
            String noiseEqualTextColorValue = cmdLine.getOptionValue('e');
            if (noiseEqualTextColorValue.equals("true") || noiseEqualTextColorValue.equals("false")) {
              config.set(ConfigConstants.NOISE_SAME_TEXT_COLOR, noiseEqualTextColorValue);
            } else {
              throw new ParseException("Only can be true or false.");
            }
          }
        }
        return config;
      }
    } catch (ParseException ex) {
      helpFormatter.printHelp("java -jar [jarfile].jar", options);
    }

    return null;
  }

}
