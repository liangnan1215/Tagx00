package trapx00.tagx00.publicdatas.mission.audio;

import trapx00.tagx00.vo.mission.audio.AudioMissionType;

import java.io.Serializable;

public class AudioJob implements Serializable {
    private AudioMissionType type;

    public AudioJob(AudioMissionType type) {
        this.type = type;
    }
}
