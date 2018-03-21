import { BaseService } from "./BaseService";
import { MissionDetail, MissionPublicItem, MissionState, MissionType } from "../models/mission/Mission";
import { ImageMissionDetail, ImageMissionType } from "../models/mission/ImageMission";
import { waitForMs } from "../../utils/Wait";

export const imgs = [
  "https://desk-fd.zol-img.com.cn/t_s960x600c5/g5/M00/0E/00/ChMkJlnJ4TOIAyeVAJqtjV-XTiAAAgzDAE7v40Amq2l708.jpg",
  "http://b1-q.mafengwo.net/s8/M00/02/BF/wKgBpVWnJL6AfCcFAA_oZSqEvVE56.jpeg?imageMogr2%2Fthumbnail%2F%21690x370r%2Fgravity%2FCenter%2Fcrop%2F%21690x370%2Fquality%2F100",
  "http://pic1.16xx8.com/allimg/170801/1-1FP116442T62.jpg",
  "http://p4-q.mafengwo.net/s7/M00/BD/0E/wKgB6lSwiyyAMQDHAA_6txvdndY07.jpeg?imageMogr2%2Fthumbnail%2F%21690x370r%2Fgravity%2FCenter%2Fcrop%2F%21690x370%2Fquality%2F100"
];



export class MissionService extends BaseService {

  constructor() {
    super("mission");
  }

  async getAllMissions() {
    // mock
    return "";
  }

  async getAMission(missionId: number) : Promise<ImageMissionDetail> {
    // mock
    return  {
      publicItem: {
        missionId: missionId,
        title: "Mission " + missionId,
        description: "Description "+missionId,
        topics: ["scenes"],
        allowCustomTag: Math.random() < 0.5,
        allowedTags: ["tag1", "tag2"],
        missionType: MissionType.IMAGE,
        start: new Date(),
        end: new Date(),
        coverUrl: "https://desk-fd.zol-img.com.cn/t_s960x600c5/g3/M0A/0F/09/Cg-4WFRplp2IYqiNACQ0TQPPChQAARbPQEM84oAJDRl464.jpg"
      },
      state: MissionState.ACTIVE,
      imageUrls: imgs,
      imageMissionTypes: [
        ImageMissionType.DISTRICT,
        ImageMissionType.PART,
        ImageMissionType.WHOLE
      ]
    } as ImageMissionDetail;
  }

}

export const missionService = new MissionService();
