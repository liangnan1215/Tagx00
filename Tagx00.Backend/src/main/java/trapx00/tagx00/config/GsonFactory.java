package trapx00.tagx00.config;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import trapx00.tagx00.config.jsonAdapter.ImageJobAdapter;
import trapx00.tagx00.config.jsonAdapter.InstanceDetailAdapter;
import trapx00.tagx00.config.jsonAdapter.MissionPropertiesAdapter;
import trapx00.tagx00.publicdatas.mission.MissionType;
import trapx00.tagx00.publicdatas.mission.image.ImageJob;
import trapx00.tagx00.vo.mission.missiontype.MissionProperties;

public class GsonFactory {
    public static Gson get() {
        return new GsonBuilder()
                .registerTypeAdapter(ImageJob.class, new ImageJobAdapter())
                .registerTypeAdapter(MissionProperties.class, new MissionPropertiesAdapter())
                .registerTypeAdapter(MissionType.class, new InstanceDetailAdapter())
                .create();
    }
}
