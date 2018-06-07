package trapx00.tagx00.entity.mission.favorite;

import trapx00.tagx00.entity.mission.AudioMission;
import trapx00.tagx00.entity.mission.ImageMission;
import trapx00.tagx00.publicdatas.mission.MissionType;

import javax.persistence.Entity;
import javax.persistence.FetchType;
import javax.persistence.JoinColumn;
import javax.persistence.ManyToOne;
import java.util.Date;

@Entity
public class AudioFavortie extends Favorite{
    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "mission_missionId")
    private AudioMission audioMission;

    public AudioFavortie() {
    }

    public AudioFavortie(String favoriteId, String workerUsername, MissionType missionType, Date acceptDate, AudioMission audioMission) {
        super(favoriteId, workerUsername, missionType, acceptDate);
        this.audioMission = audioMission;
    }

    public AudioMission getAudioMission() {
        return audioMission;
    }

    public void setAudioMission(AudioMission audioMission) {
        this.audioMission = audioMission;
    }
}
