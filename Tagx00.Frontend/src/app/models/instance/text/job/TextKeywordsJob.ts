import { TextJob } from "./TextJob";
import { TagTuple } from "../../TagTuple";
import { TextMissionType } from "../../../mission/text/TextMissionProperties";

export interface TextKeywordsJob extends TextJob {
  type: TextMissionType.KEYWORDS;
  tagTuples: TagTuple[];
}
