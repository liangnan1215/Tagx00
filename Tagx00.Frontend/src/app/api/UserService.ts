import { HttpService, NetworkResponse } from "./HttpService";
import { HttpMethod } from "./utils";
import { Inject, Injectable } from "react.di";
import { UserRole } from "../models/User";

export interface LoginResult {
  token: string,
  jwtRoles: { authority: string }[];
  email: string;
}

export interface UserRegisterResponse {
  token: string
}

export interface UserRegisterConfirmationResponse {
  token: string
  jwtRoles: string[]
  email: string
}

function encryptPassword(password: string) {
  return password;
}

@Injectable
export class UserService {

  constructor(@Inject private http: HttpService) {
  }

  async login(username: string, password: string): Promise<LoginResult> {
    password = encryptPassword(password);

    const res = await this.http.fetch({
      path: "account/login",
      queryParams: {username, password}
    });
    return res.response;
  }

  async register(username: string, password: string, email: string, role: UserRole): Promise<NetworkResponse<UserRegisterResponse>> {
    console.log(username)
    password = encryptPassword(password);
    return await this.http.fetch({
      path: "account/register",
      queryParams: {username, password, email, role},
      method: HttpMethod.POST
    });
  }

  async registerValidate(token: string, code: string): Promise<UserRegisterConfirmationResponse> {
    const res = await this.http.fetch({
      path: "account/register/validate",
      body: {token, code},
      method: HttpMethod.POST
    });
    return res.response;
  }

}
