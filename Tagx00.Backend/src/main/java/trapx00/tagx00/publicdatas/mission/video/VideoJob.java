package trapx00.tagx00.publicdatas.mission.video;

import trapx00.tagx00.vo.mission.video.VideoMissionType;

import java.io.Serializable;

public class VideoJob implements Serializable {

    private VideoMissionType type;

    public VideoJob() {
    }

    public VideoJob(VideoMissionType type) {
        this.type = type;
    }

    public VideoMissionType getType() {
        return type;
    }

    public void setType(VideoMissionType type) {
        this.type = type;
    }




}
