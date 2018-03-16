package trapx00.tagx00.dataservice.mission;

import trapx00.tagx00.vo.mission.instance.MissionInstanceItemVo;
import trapx00.tagx00.vo.mission.missiontype.MissionVo;
import trapx00.tagx00.vo.mission.requester.MissionRequesterQueryItemVo;

public interface RequesterMissionDataService {

    /**
     * save mission
     * @param missionVo
     */
    void saveMission (MissionVo missionVo);

    /**
     * get missionid by username
     * @param username
     * @return the list of  the MissionRequesterQueryItemVo
     */
    MissionRequesterQueryItemVo[] getMissionByUsername(String username);

    /**
     * get instance by instanceId
     * @param instanceId
     * @return the specific MissionInstanceItemVo
     */
    MissionInstanceItemVo getInstanceById(int  instanceId);


    /**
     * get all instances of the user by username
     * @param username
     * @return the list of missionIstanceItemVo
     */
    MissionInstanceItemVo[] getInstanceByUsername(String username);

    /**
     * get the instance by username and missionId
     * @param username
     * @param missionId
     * @return the instance matching username and missionId
     */
    MissionInstanceItemVo getInstanceByUsernameAndMissionId(String username,int missionId);

}