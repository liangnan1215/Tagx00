package trapx00.tagx00.data.account;

import trapx00.tagx00.dataservice.account.RequesterInfoDataService;
import trapx00.tagx00.entity.account.User;
import trapx00.tagx00.entity.mission.Instance;
import trapx00.tagx00.entity.mission.Mission;

public class RequesterInfoDataServiceImpl implements RequesterInfoDataService {

    /**
     * get user by username
     * @param Username
     * @return
     */
    @Override
    public User getUserByUsername(String Username) {
        return null;
    }
    /**
     * get missions by requesterUsername
     * @param requesterUsername
     * @return
     */
    @Override
    public Mission[] getMissionsByRequesterUsername(String requesterUsername) {
        return new Mission[0];
    }
    /**
     * get instances by missionId
     * @param missionId
     * @return
     */
    @Override
    public Instance[] getInstancesByMissionId(int missionId) {
        return new Instance[0];
    }
}
