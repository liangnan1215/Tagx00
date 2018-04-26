import { MissionInstanceState } from "./MissionInstanceState";

export interface Instance {

  instanceId: string;

  workerUsername: string;

  missionInstanceState: MissionInstanceState;

  missionId: string;

  acceptDate: Date;

  submitDate: Date;

  isSubmitted: boolean;

  completedJobsCount: number;

  expRatio?: number;
  exp?: number;
  credits?: number;
  comment?: string;
}

export function convertToInstance(json: any) {

}
