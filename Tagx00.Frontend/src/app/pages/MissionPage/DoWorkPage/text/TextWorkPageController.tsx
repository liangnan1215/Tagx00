import { TextMissionDetail } from "../../../../models/mission/text/TextMissionDetail";
import { TextInstanceDetail } from "../../../../models/instance/text/TextInstanceDetail";
import { TextJob } from "../../../../models/instance/text/job/TextJob";
import { MissionType } from "../../../../models/mission/Mission";
import { TextMissionSetting, TextMissionType } from "../../../../models/mission/text/TextMissionProperties";
import { TextKeywordsJob } from "../../../../models/instance/text/job/TextKeywordsJob";
import { TextClassificationJob } from "../../../../models/instance/text/job/TextClassificationJob";
import { TextResult } from "../../../../models/instance/text/TextResult";
import { Notation, WorkPageController } from "../WorkPageController";


export interface TextNotation<T extends TextJob> extends Notation<T> {
  textUrl: string;
  job: T;
}

type KnownTextJob = TextKeywordsJob | TextClassificationJob;

function any<T>(array: T[]) {
  return !!array && array.length > 0;
}

function judgeJobComplete(job: KnownTextJob) {
  if (!job) return false;
  switch (job.type) {
    case TextMissionType.CLASSIFICATION:
      return any(job.tagTuples);
    case TextMissionType.KEYWORDS:
      return any(job.tagTuples);
  }
  return false;
}

export class TextWorkPageController extends WorkPageController<TextMissionDetail, TextInstanceDetail, TextJob, TextNotation<TextJob>> {

  textUrls: string[] = [];

  settings: TextMissionSetting[];

  currentInstanceDetail(): TextInstanceDetail {
    const {instance} = this.initialDetail;
    return {
      missionType: MissionType.TEXT,
      textResults: this.currentNotations.map((x, index) => ({
        workResultId: index+"",
        textJob: x.job,
        url: x.textUrl,
        isDone: judgeJobComplete(x.job as any)
      }) as TextResult),
      instance: instance
    }
  }

  constructor(missionDetail: TextMissionDetail, instanceDetail: TextInstanceDetail) {
    super(missionDetail, instanceDetail);
    this.settings = missionDetail.settings;
    this.textUrls = missionDetail.textUrls;

    // initialize jobs
    for (const url of missionDetail.textUrls) {
      for (const setting of missionDetail.settings) {
        // find if the result is already exists
        const result = instanceDetail.textResults
          && instanceDetail.textResults.find(x => x.url == url && x.textJob.type == setting.textMissionType);

        if (!result){
          this.currentNotations.push({
            textUrl: url,
            job: { type: setting.textMissionType }
          });
        } else {
          this.currentNotations.push({
            textUrl: url,
            job: result.textJob
          });
        }
      }
    }
  }





}
