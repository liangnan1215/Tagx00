package trapx00.tagx00.dataservice.mission;

import trapx00.tagx00.entity.mission.instance.Instance;
import trapx00.tagx00.exception.viewexception.MissionAlreadyAcceptedException;
import trapx00.tagx00.exception.viewexception.SystemException;
import trapx00.tagx00.publicdatas.mission.MissionType;
import trapx00.tagx00.vo.mission.instance.InstanceDetailVo;
import trapx00.tagx00.vo.mission.instance.InstanceVo;

public interface WorkerMissionDataService {

    /**
     * save the progress of the instance.
     * if not accpet the mission before, the system will create a instance for workers
     * also use to abort the instance
     *
     * @param instanceVo
     */
    int saveInstanceDetailVo(InstanceDetailVo instanceVo) throws SystemException, MissionAlreadyAcceptedException;

    /**
     * save the instance
     *
     * @param instanceId
     * @param missionType
     */
    int abortInstance(int instanceId, MissionType missionType);


    /**
     * get instance by username
     *
     * @param workerUsername
     * @return the list of  the MissionWorkerQueryItemVo
     */
    InstanceVo[] getInstanceByWorkerUsername(String workerUsername);

    /**
     * get the information of  instance by username and missionId
     *
     * @param workerUsername
     * @param missionId
     * @param missionType
     * @return the instance matching username and missionId
     */
    InstanceDetailVo getInstanceDetailVoByUsernameAndMissionId(String workerUsername, int missionId, MissionType missionType);

    /**
     * get the information of  instance by username and missionId
     *
     * @param workerUsername
     * @param missionId
     * @param missionType
     * @return the instance matching username and missionId
     */
    Instance getInstanceByUsernameAndMissionId(String workerUsername, int missionId, MissionType missionType);

    /**
     * delte the mission of a worker
     *
     * @param missionId
     * @param username
     * @param missionType
     * @return
     */
    boolean deleteInstanceByMissionIdAndUsername(int missionId, String username, MissionType missionType);
}
