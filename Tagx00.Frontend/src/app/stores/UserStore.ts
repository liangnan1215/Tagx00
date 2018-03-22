import { action, computed, observable } from "mobx";
import { User, UserRole } from "../models/User";
import { STORE_USER } from "../constants/stores";
import { LoginResult } from "../api/UserService";
import { localStorage } from './UiUtil';

export class UserStore {
  @observable user: User = null;



  @computed get loggedIn() {
    return !!this.user;
  }

  get token() {
    return this.user ? this.user.token : null;
  }

  @computed get isAdmin() {
    return this.user && this.user.role === UserRole.Admin;
  }

  @action logout() {
    this.user = null;
    this.clearUser();
  };


  @action async login(response: LoginResult) {
    this.user = new User(response);
  };

  remember() {
    localStorage.setItem("user", JSON.stringify(this.user));
  }

  clearUser() {
    localStorage.removeItem("user");
  }

  constructor(detectLocalStorage: boolean = true) {
    if (detectLocalStorage) {
      const user = localStorage.getItem("user");
      if (user) {
        try {
          this.user = new User(JSON.parse(user));
        } catch (ignored) {
          console.log(ignored);
        }
      }
    }
  }
}

export interface UserStoreProps {
  [STORE_USER]?: UserStore
}
