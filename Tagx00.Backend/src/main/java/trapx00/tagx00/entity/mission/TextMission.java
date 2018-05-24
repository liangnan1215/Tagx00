package trapx00.tagx00.entity.mission;


import trapx00.tagx00.entity.mission.favorite.TextFavorite;
import trapx00.tagx00.entity.mission.instance.TextInstance;
import trapx00.tagx00.publicdatas.mission.MissionState;
import trapx00.tagx00.publicdatas.mission.MissionType;
import trapx00.tagx00.vo.mission.text.TextMissionType;

import javax.persistence.*;
import java.util.Date;
import java.util.List;

@Entity
public class TextMission extends Mission {
    @Column(name = "textUrls")
    @ElementCollection(targetClass = String.class)
    private List<String> textUrls;
    @Column(name = "textMissionType")
    @ElementCollection(targetClass = TextMissionType.class)
    private List<TextMissionType> textMissionTypes;
    @OneToMany(mappedBy = "textMission", cascade = {CascadeType.PERSIST, CascadeType.MERGE}, fetch = FetchType.LAZY)
    private List<TextInstance> textInstances;
    @OneToMany(mappedBy = "textMission", cascade = {CascadeType.PERSIST, CascadeType.MERGE}, fetch = FetchType.LAZY)
    private List<TextFavorite> textFavorites;

    public TextMission() {
    }

    public TextMission(String missionId, String title, String description,
                       List<String> topics, MissionType missionType,
                       MissionState missionState, Date start, Date end,
                       String coverUrl, String requesterUsername, int level,
                       int credits, int minimalWorkerLevel, List<String> browserUsers, List<String> textUrls,
                       List<TextMissionType> textMissionTypes,
                       List<TextInstance> textInstances, List<TextFavorite> textFavorites) {
        super(missionId, title, description, topics, missionType, missionState, start, end, coverUrl, requesterUsername, level, credits, minimalWorkerLevel, browserUsers);
        this.textUrls = textUrls;
        this.textMissionTypes = textMissionTypes;
        this.textInstances = textInstances;
        this.textFavorites = textFavorites;
    }

    public List<String> getTextUrls() {
        return textUrls;
    }

    public void setTextUrls(List<String> textUrls) {
        this.textUrls = textUrls;
    }

    public List<TextMissionType> getTextMissionTypes() {
        return textMissionTypes;
    }

    public void setTextMissionTypes(List<TextMissionType> textMissionTypes) {
        this.textMissionTypes = textMissionTypes;
    }

    public List<TextInstance> getTextInstances() {
        return textInstances;
    }

    public void setTextInstances(List<TextInstance> textInstances) {
        this.textInstances = textInstances;
    }

    public List<TextFavorite> getTextFavorites() {
        return textFavorites;
    }

    public void setTextFavorites(List<TextFavorite> textFavorites) {
        this.textFavorites = textFavorites;
    }
}
