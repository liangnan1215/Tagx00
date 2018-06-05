package trapx00.tagx00.datacollect;

import com.google.gson.Gson;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import trapx00.tagx00.data.dao.mission.instance.ImageInstanceDao;
import trapx00.tagx00.entity.mission.MissionAsset;
import trapx00.tagx00.entity.mission.instance.ImageInstance;
import trapx00.tagx00.entity.mission.instance.workresult.ImageResult;
import trapx00.tagx00.publicdatas.mission.TagTuple;
import trapx00.tagx00.publicdatas.mission.image.whole.ImageWholeJob;
import trapx00.tagx00.util.PathUtil;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

@Service
public class DataCollectorServiceImpl implements DataCollectorService {
    private final ImageInstanceDao imageInstanceDao;

    @Autowired
    public DataCollectorServiceImpl(ImageInstanceDao imageInstanceDao) {
        this.imageInstanceDao = imageInstanceDao;
    }


    @Override
    public void collectData() {
        List<ImageInstance> imageInstanceList = imageInstanceDao.findAll();
        try {
            File file = new File("/Users/apple/Documents/workspace/java/SE3/Tagx00.MachineLearning/data/proval/train.txt");
            BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file, true)));
            for (ImageInstance imageInstance : imageInstanceList) {
                ImageInstance imageInstanceWithResults = getImageInstance(imageInstance.getInstanceId());
                List<MissionAsset> missionAssets = imageInstanceWithResults.getImageMission().getMissionAssets();
                for (int i = 0; i < missionAssets.size(); i++) {
                    List<TagTuple> tagTuples = ((ImageWholeJob) imageInstanceWithResults.getImageResults().get(i).getImageJob()).getTuple().getTagTuples();
                    List<String> tags = tagTuples.stream().collect(ArrayList::new, (list, tagTuple) -> list.add(tagTuple.getTag()), ArrayList::addAll);
                    DataObject dataObject = new DataObject(missionAssets.get(i).getUrl(), tags, missionAssets.get(i).getTagConfTuple());
                    out.write(new Gson().toJson(dataObject));
                }
            }
            out.flush();
            out.close();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    private ImageInstance getImageInstance(String instanceId) throws IOException, ClassNotFoundException {
        ImageInstance imageInstance = imageInstanceDao.findImageInstanceByInstanceId(instanceId);
        FileInputStream fileIn = new FileInputStream(PathUtil.getSerPath() + "image_instance" + "_" + instanceId);
        ObjectInputStream in = new ObjectInputStream(fileIn);
        List<ImageResult> imageResults = (List<ImageResult>) in.readObject();
        in.close();
        fileIn.close();
        imageInstance.setImageResults(imageResults);
        return imageInstance;
    }
}
