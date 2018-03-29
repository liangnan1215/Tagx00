package trapx00.tagx00.util;

import trapx00.tagx00.MainApplication;

import java.net.URL;

public class ResourceUtil {
    public static URL getResource(String path) {
        return MainApplication.class.getResource(path);
    }

    public static String getFilePathUnderRootDirOfJarFileOrClassDir(String relativePath) {
        String path = MainApplication.class.getProtectionDomain().getCodeSource().getLocation().getPath();
//        String path = MainApplication.class.getProtectionDomain().getCodeSource().getLocation().getPath();
//        String[] paths = path.split("!");
//        String filePath = paths[0];
//
//        File jarFile = new File(paths[0]);
//        String parentDir = Arrays.stream(jarFile.getParent().split(":"))
//                .skip(1).collect(Collectors.joining(":"));
//
//
//        System.out.println(parentDir);
//        return parentDir + relativePath;

        return path + ".." + relativePath;
    }
}
