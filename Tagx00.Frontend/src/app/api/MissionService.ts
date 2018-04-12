import { HttpService } from "./HttpService";
import { MissionPublicItem } from "../models/mission/Mission";
import { ImageMissionDetail } from "../models/mission/image/ImageMission";
import { HttpMethod } from "./utils";
import { Inject, Injectable } from "react.di";
import { Topic } from "../models/topic/Topic";

export interface TopicFetchResponse {
  topics: Topic[]
}

@Injectable
export class MissionService {

  constructor(@Inject private http: HttpService) {
  }

  async getAllMissions(): Promise<MissionPublicItem[]> {
    const res = await this.http.fetch({
      path: "/mission",
      method: HttpMethod.GET
    });
    return res.response.items as MissionPublicItem[];

  }

  async getAMission(missionId: number | string, token: string): Promise<ImageMissionDetail> {
    const res = await this.http.fetch({
      path: `/mission/${missionId}`,
      token: token
    });

    if (res.ok) {
      return res.response.detail;
    } else {
      throw res.error;
    }
  }

  async getAllTopics(): Promise<TopicFetchResponse> {
    const res = await this.http.fetch({
      path: "/mission/topics",
      method: HttpMethod.GET
    });
    return res.response;
  }

}
