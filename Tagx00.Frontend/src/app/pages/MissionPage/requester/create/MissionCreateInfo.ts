import { MissionCreate } from "../../../../models/mission/create/MissionCreate";
import { MissionType } from "../../../../models/mission/Mission";
import { UploadFile } from "antd/lib/upload/interface";
import { computed, observable, toJS } from "mobx";
import { ImageMissionProperties } from "../../../../models/mission/image/ImageMissionProperties";
import { ImageMissionType } from "../../../../models/mission/image/ImageMission";
import moment from 'moment';

export class MissionCreateInfo {
  @observable title: string = "";
  @observable description: string = "";
  @observable remainingCredits = -1;
  @observable imageMissionTypes: ImageMissionType[] = [];
  @observable images: UploadFile[] = [];

  @observable topics: string[] = [];
  @observable dateRange: [moment.Moment, moment.Moment] = [null,null];
  @observable level: string = "1";
  @observable minimalWorkerLevel: string = "1";
  @observable credits: number = 0;
  @observable creditsValid: boolean = true;
  @observable coverImage: UploadFile = null;

  @observable createAttempted: boolean = false;

  get missionCreateVo(): MissionCreate {
    return toJS({
      title: this.title,
      description: this.description,
      topics: this.topics,
      properties: null,
      start: this.dateRange[0].toDate(),
      end: this.dateRange[1].toDate(),
      minimalWorkerLevel: parseInt(this.minimalWorkerLevel),
      level: parseInt(this.level),
      credits: this.credits,
    });
  }

  @computed get titleValid() {
    return !this.createAttempted || !!this.title;
  }

  @computed get descriptionValid() {
    return !this.createAttempted || !!this.description;
  }

  @computed get typesValid() {
    return !this.createAttempted || this.imageMissionTypes.length >0;
  }

  @computed get dateRangeValid() {
    return !this.createAttempted || this.dateRange[0] != null;
  }

  @computed get imageTypesValid() {
    return !this.createAttempted || this.imageMissionTypes.length >0;
  }

  @computed get imagesValid() {
    return !this.createAttempted || this.images.length > 0;
  }

  @computed get minimalWorkerLevelValid() {
    const parsed = parseInt(this.minimalWorkerLevel);
    if (isNaN(parsed)) return false;
    return 1<=parsed && parsed <=100;
  }

  @computed get levelValid() {
    const parsed = parseInt(this.level);
    if (isNaN(parsed)) return false;
    return 1<=parsed && parsed <=5;
  }


  @computed get valid() {
    return this.titleValid && this.descriptionValid
      && this.typesValid
      && this.dateRangeValid && this.imageTypesValid
      && this.imagesValid && this.minimalWorkerLevelValid && this.levelValid
      && this.creditsValid;
  }
}
