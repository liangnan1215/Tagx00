import React from "react";
import { AsyncRouteConfig, RedirectRouteConfig, RouteType } from "../../router/RouteConfig";
export const missionPage: AsyncRouteConfig = {
  type: RouteType.Async,
  exact: false,
  path: "/mission",
  render: async (props) => {
    const Page = (await import("../MissionPage")).MissionPage;
    return <Page/>;
  },
};

export const browsePage: AsyncRouteConfig  = {
  type: RouteType.Async,
  exact: true,
  path: "/browse",
  render: async (props) => {
    const BrowsePage = (await import("../BrowsePage")).BrowsePage;
    return <BrowsePage/>;
  },
};

export const registerPage: AsyncRouteConfig = {
  type: RouteType.Async,
  exact: true,
  path: "/register",
  render: async (props) => {
    const RegisterPage = (await import("../../pages/RegisterPage")).RegisterPage;
    return <RegisterPage/>;
  },
};

export const aboutPage:AsyncRouteConfig = {
  type: RouteType.Async,
  exact: true,
  path: "/about",
  render: async (props) => {
    const AboutPage = (await import("../AboutPage")).AboutPage;
    return <AboutPage/>;
  },
};

export const selfCenterPage: AsyncRouteConfig = {
  type: RouteType.Async,
  exact: false,
  path: "/self",
  render: async (props) => {
    const Page = (await import("../SelfPage")).SelfPage;
    return <Page/>;
  }
};





export default [
  missionPage,
  browsePage,
  registerPage,
  aboutPage,
  selfCenterPage
]
