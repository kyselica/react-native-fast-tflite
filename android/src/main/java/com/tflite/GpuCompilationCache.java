package com.tflite;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

/** Only compiled programs live here; never model paths, user data or live interpreters. */
public final class GpuCompilationCache {
  private GpuCompilationCache() {}

  public static String directory(File codeCache, ByteBuffer model, String configuration) {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      digest.update(configuration.getBytes(StandardCharsets.UTF_8));
      digest.update((byte)0);
      digest.update(model);
      byte[] hash = digest.digest();
      StringBuilder key = new StringBuilder();
      for (byte value : hash)
        key.append(String.format("%02x", value & 0xff));
      File root = new File(codeCache, "fast-tflite-gpu-v1");
      if ((!root.isDirectory() && !root.mkdirs()) || !root.canWrite())
        return "";
      return new File(root, key.toString()).getAbsolutePath();
    } catch (Exception ignored) {
      return "";
    }
  }
}
