package trapx00.tagx00.dataservice.account;

import trapx00.tagx00.entity.mission.Mission;
import trapx00.tagx00.entity.mission.instance.Instance;

public interface RequesterInfoDataService {

    /**
     * get missions by requesterUsername
     *
     * @param requesterUsername
     * @return
     */
    Mission[] getMissionsByRequesterUsername(String requesterUsername);

    /**
     * get instances by missionId
     *
     * @param missionId
     * @return
     */
    Instance[] getInstancesByMissionId(int missionId);
}
