package me.zouzhipeng;

import java.util.concurrent.Executor;

public class PoolExecutor implements Executor  {

  @Override
  public void execute(Runnable command) {
    if (null == command) {
      return;
    }
    command.run();
  }

}