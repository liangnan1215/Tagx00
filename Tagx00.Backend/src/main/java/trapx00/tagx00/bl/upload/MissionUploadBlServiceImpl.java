package trapx00.tagx00.bl.upload;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import trapx00.tagx00.blservice.upload.MissionUploadBlService;
import trapx00.tagx00.dataservice.mission.RequesterMissionDataService;
import trapx00.tagx00.dataservice.upload.ImageDataService;
import trapx00.tagx00.dataservice.upload.TextDataService;
import trapx00.tagx00.entity.mission.ImageMission;
import trapx00.tagx00.entity.mission.TextMission;
import trapx00.tagx00.exception.viewexception.MissionIdDoesNotExistException;
import trapx00.tagx00.exception.viewexception.SystemException;
import trapx00.tagx00.response.upload.UploadMissionImageResponse;
import trapx00.tagx00.response.upload.UploadMissionTextResponse;
import trapx00.tagx00.util.MissionUtil;
import trapx00.tagx00.util.PathUtil;

import javax.imageio.stream.FileImageOutputStream;
import java.io.*;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;


@Service
public class MissionUploadBlServiceImpl implements MissionUploadBlService {

    private final ImageDataService imageDataService;
    private final TextDataService textDataService;
    private final RequesterMissionDataService requesterMissionDataService;
    private final static String TEMP_PATH = PathUtil.getTmpPath();

    @Autowired
    public MissionUploadBlServiceImpl(ImageDataService imageDataService, TextDataService textDataService, RequesterMissionDataService requesterMissionDataService) {
        this.imageDataService = imageDataService;
        this.textDataService = textDataService;
        this.requesterMissionDataService = requesterMissionDataService;
    }

    /**
     * Upload the image of the mission
     *
     * @param missionId
     * @param multipartFile
     * @param order
     * @param isCover
     * @return the url of the image
     */
    @Override
    public UploadMissionImageResponse uploadImage(String missionId, MultipartFile multipartFile, int order, boolean isCover) throws SystemException, MissionIdDoesNotExistException {
        try {
            switch (MissionUtil.getType(missionId)) {
                case TEXT:
                    TextMission textMission = (TextMission) requesterMissionDataService.getMissionByMissionId(missionId);
                    String url = imageDataService.uploadImage(generateImageKey(missionId, order, isCover), multipartFile.getBytes());
                    if (isCover) {
                        textMission.setCoverUrl(url);
                    }
                    requesterMissionDataService.updateMission(textMission);
                    return new UploadMissionImageResponse(url);
                case IMAGE:
                    ImageMission imageMission = (ImageMission) requesterMissionDataService.getMissionByMissionId(missionId);
                    url = imageDataService.uploadImage(generateImageKey(missionId, order, isCover), multipartFile.getBytes());
                    List<String> urls = imageMission.getImageUrls();
                    if (isCover) {
                        imageMission.setCoverUrl(url);
                    } else {
                        urls.add(url);
                        imageMission.setImageUrls(urls);
                    }
                    requesterMissionDataService.updateMission(imageMission);
                    return new UploadMissionImageResponse(url);
            }
        } catch (IOException e) {
            e.printStackTrace();
            throw new SystemException();
        }
        return null;
    }

    /**
     * Upload the text of the mission
     *
     * @param missionId
     * @param multipartFile
     * @return the urls of the texts
     */
    @Override
    public UploadMissionTextResponse uploadText(String missionId, MultipartFile multipartFile) throws SystemException, MissionIdDoesNotExistException {
        //保存到临时文件
        try {
            TextMission textMission = (TextMission) requesterMissionDataService.getMissionByMissionId(missionId);
            List<String> textUrls = new ArrayList<>();
            File file = new File(TEMP_PATH + "/text");
            FileImageOutputStream fileWriter = new FileImageOutputStream(file);
            fileWriter.write(multipartFile.getBytes());
            fileWriter.close();

            String descDir = TEMP_PATH + "/textZip";
            File pathFile = new File(descDir);
            if (!pathFile.exists()) {
                pathFile.mkdirs();
            }
            ZipFile zip = new ZipFile(file);
            int index = 0;
            for (Enumeration entries = zip.entries(); entries.hasMoreElements(); ) {
                ZipEntry entry = (ZipEntry) entries.nextElement();
                String zipEntryName = entry.getName();
                InputStream in = zip.getInputStream(entry);
                String outPath = (descDir + "/" + zipEntryName).replaceAll("\\*", "/");

                //判断路径是否存在,不存在则创建文件路径
                file = new File(outPath.substring(0, outPath.lastIndexOf('/')));
                if (!file.exists()) {
                    file.mkdirs();
                }
                //判断文件全路径是否为文件夹,如果是上面已经上传,不需要解压
                if (new File(outPath).isDirectory()) {
                    continue;
                }
                //输出文件路径信息
                System.out.println(outPath);

                OutputStream out = new FileOutputStream(outPath);
                byte[] buf1 = new byte[1024];
                int len;
                while ((len = in.read(buf1)) > 0) {
                    out.write(buf1, 0, len);
                }
                in.close();
                out.close();

                File textFile = new File(outPath);
                Long textFileLength = textFile.length();
                byte[] textFileContent = new byte[textFileLength.intValue()];
                in = new FileInputStream(file);
                in.read(textFileContent);
                in.close();
                String url = textDataService.uploadText(generateTextKey(missionId, index), new String(textFileContent, "utf-8"));
                textUrls.add(url);

                index++;
            }
            textMission.setTextUrls(textUrls);
            requesterMissionDataService.updateMission(textMission);
            System.out.println("******************解压完毕********************");
            return new UploadMissionTextResponse("success");
        } catch (IOException e) {
            e.printStackTrace();
            throw new SystemException();
        }
    }

    private String generateImageKey(String missionId, int order, boolean isCover) {
        if (isCover) {
            return "image_cover" + missionId;
        } else {
            return "image_" + missionId + "_" + order;
        }
    }

    private String generateTextKey(String missionId, int order) {
        return "text_" + missionId + "_" + order;
    }
}
