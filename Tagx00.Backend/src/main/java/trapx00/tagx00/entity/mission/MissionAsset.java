package trapx00.tagx00.entity.mission;

import javax.persistence.Embeddable;
import java.util.Map;

@Embeddable
public class MissionAsset {
    private String url;
    private Map<String, Double> TagConfTuple;

    public MissionAsset() {
    }

    public MissionAsset(String url, Map<String, Double> tagConfTuple) {
        this.url = url;
        TagConfTuple = tagConfTuple;
    }

    public String getUrl() {
        return url;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public Map<String, Double> getTagConfTuple() {
        return TagConfTuple;
    }

    public void setTagConfTuple(Map<String, Double> tagConfTuple) {
        TagConfTuple = tagConfTuple;
    }
}
