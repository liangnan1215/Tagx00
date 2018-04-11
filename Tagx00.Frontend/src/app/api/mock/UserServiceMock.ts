import { Injectable } from "react.di";
import { NetworkResponse } from "../HttpService";
import { LoginResult, UserRegisterConfirmationResponse, UserRegisterResponse, UserService } from "../UserService";
import { UserRole } from "../../models/User";
import { HttpMethod } from "../utils";

@Injectable
export class UserServiceMock extends UserService {

  async login(username: string, password: string): Promise<NetworkResponse<LoginResult>> {

    return new NetworkResponse(200, {
        token: "123",
        jwtRoles: [{ authority: "ROLE_REQUESTER"}],
        email: "1@1.com"
      }
    );
  }

  async register(username: string, password: string): Promise<NetworkResponse<UserRegisterResponse>> {
    return new NetworkResponse(201, {
        token: "123",
      }
    );
  }

  async registerValidate(token: string, code: string): Promise<NetworkResponse<UserRegisterConfirmationResponse>> {

    return new NetworkResponse(200, {
        token: "123",
        jwtRoles: [{ authority: "ROLE_REQUESTER"}],
        email: "1@1.com"
      }
    );
  }

}
