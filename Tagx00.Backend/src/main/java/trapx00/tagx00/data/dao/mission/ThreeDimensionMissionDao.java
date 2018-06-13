package trapx00.tagx00.data.dao.mission;

import org.springframework.data.jpa.repository.JpaRepository;
import trapx00.tagx00.entity.mission.ThreeDimensionMission;

import java.util.ArrayList;

public interface ThreeDimensionMissionDao extends JpaRepository<ThreeDimensionMission, String> {

    ThreeDimensionMission findTHreeDimensionMissionByMissionId(String missionId);

    ArrayList<ThreeDimensionMission> findThreeDimensionMissionsByRequesterUsername(String requesterUsername);
}
