package trapx00.tagx00.vo.mission.image;

import trapx00.tagx00.publicdatas.mission.MissionState;
import trapx00.tagx00.publicdatas.mission.MissionType;
import trapx00.tagx00.vo.mission.forpublic.MissionDetailVo;
import trapx00.tagx00.vo.mission.forpublic.MissionPublicItemVo;

import java.util.List;

public class ImageMissionDetailVo extends MissionDetailVo {

    private List<String> imageUrls;

    private List<ImageMissionType> imageMissionTypes;

    public ImageMissionDetailVo() {
    }

    public ImageMissionDetailVo(MissionPublicItemVo publicItem, MissionState missionState, String requesterUsername, MissionType missionType, List<String> imageUrls, List<ImageMissionType> imageMissionTypes) {
        super(publicItem, missionState, requesterUsername, missionType);
        this.imageUrls = imageUrls;
        this.imageMissionTypes = imageMissionTypes;
    }

    public List<String> getImageUrls() {
        return imageUrls;
    }

    public void setImageUrls(List<String> imageUrls) {
        this.imageUrls = imageUrls;
    }

    public List<ImageMissionType> getImageMissionTypes() {
        return imageMissionTypes;
    }

    public void setImageMissionTypes(List<ImageMissionType> imageMissionTypes) {
        this.imageMissionTypes = imageMissionTypes;
    }
}
