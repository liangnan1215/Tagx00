package trapx00.tagx00.data.dao.mission;

import org.springframework.data.jpa.repository.JpaRepository;
import trapx00.tagx00.entity.mission.ImageMission;

import java.util.ArrayList;

public interface ImageMissionDao extends JpaRepository<ImageMission, Integer> {
    ImageMission findImageMissionByMissionId(int missionId);

    ArrayList<ImageMission> findImageMissionsByRequesterUsername(String requesterUsername);
}
