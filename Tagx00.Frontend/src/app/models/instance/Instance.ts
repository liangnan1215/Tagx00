import { MissionInstanceState } from "./MissionInstanceState";

export interface Instance {

  instanceId: number;

  workerUsername: string;

  missionInstanceState: MissionInstanceState;

  missionId: number;

  acceptDate: Date;

  submitDate: Date;

  isSubmitted: boolean;

  completedJobsCount: number;
}

export function convertToInstance(json: any) {

}
