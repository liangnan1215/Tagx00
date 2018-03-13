package trapx00.tagx00.bl.user;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;
import trapx00.tagx00.blservice.user.UserBlService;
import trapx00.tagx00.dataservice.user.UserDataService;
import trapx00.tagx00.entity.user.Role;
import trapx00.tagx00.entity.user.User;
import trapx00.tagx00.exception.viewexception.SystemException;
import trapx00.tagx00.exception.viewexception.UserAlreadyExistsException;
import trapx00.tagx00.exception.viewexception.WrongUsernameOrPasswordException;
import trapx00.tagx00.response.user.UserLoginResponse;
import trapx00.tagx00.response.user.UserRegisterResponse;
import trapx00.tagx00.security.jwt.JwtRole;
import trapx00.tagx00.security.jwt.JwtService;
import trapx00.tagx00.security.jwt.JwtUser;
import trapx00.tagx00.util.Convertor;
import trapx00.tagx00.vo.user.UserSaveVo;

import java.util.Collection;

@Service
public class UserBlServiceImpl implements UserBlService {

    private final UserDataService userDataService;
    private final UserDetailsService userDetailsService;
    private final JwtService jwtService;

    @Autowired
    public UserBlServiceImpl(UserDataService userDataService, UserDetailsService userDetailsService, JwtService jwtService) {
        this.userDataService = userDataService;
        this.userDetailsService = userDetailsService;
        this.jwtService = jwtService;
    }

    /**
     * sign up
     *
     * @param userSaveVo the user to be registered
     */
    @Override
    public UserRegisterResponse signUp(UserSaveVo userSaveVo) throws UserAlreadyExistsException, SystemException {
        if (userDataService.isUserExistent(userSaveVo.getUsername())) {
            throw new UserAlreadyExistsException();
        } else {
            BCryptPasswordEncoder encoder = new BCryptPasswordEncoder();
            final String rawPassword = userSaveVo.getPassword();
            userSaveVo.setPassword(encoder.encode(rawPassword));

            User user = Convertor.userSaveVoToUser(userSaveVo);
            JwtUser jwtUser = jwtService.convertUserToJwtUser(user);
            userDataService.saveUser(user);
            String token = jwtService.generateToken(jwtUser);
            String email = jwtUser.getEmail();
            Collection<JwtRole> jwtRoles = jwtUser.getAuthorities();
            return new UserRegisterResponse(token, jwtRoles, email);
        }
    }

    @Override
    public UserLoginResponse login(String username, String password) throws WrongUsernameOrPasswordException {
        if (userDataService.confirmPassword(username, password)) {
            JwtUser jwtUser = (JwtUser) userDetailsService.loadUserByUsername(username);
            String token = jwtService.generateToken(jwtUser);
            String email = jwtUser.getEmail();
            Collection<JwtRole> jwtRoles = jwtUser.getAuthorities();
            return new UserLoginResponse(token, jwtRoles, email);
        } else {
            throw new WrongUsernameOrPasswordException();
        }
    }
}
