package trapx00.tagx00.security.jwt;

import io.jsonwebtoken.Claims;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;
import trapx00.tagx00.entity.user.Role;
import trapx00.tagx00.springcontroller.user.UserController;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;

import static org.junit.Assert.assertEquals;


@RunWith(SpringRunner.class)
@SpringBootTest
public class JwtServiceTest {
    @Value("${jwt.claimKey.username}")
    private String claimKeyUsername;

    @Value("${jwt.claimKey.authorities}")
    private String claimKeyAuthorities;

    @Value("${jwt.secret}")
    private String secret;

    @Value("${jwt.expiration}")
    private long expiration;

    @Autowired
    private JwtService service;

    private JwtUser user = new JwtUser("test","test","test@test.com", Arrays.asList(JwtRole.WORKER));

    @Test
    public void convertUserToJwtUser() {
    }

    @Test
    public void generateToken() {

    }

    public String getToken() {
        return service.generateToken(user);
    }


    @Test
    public void getClaimsFromToken() {
        String token = getToken();
        Claims claimsFromToken = service.getClaimsFromToken(token);
        assertEquals("test", claimsFromToken.get(claimKeyUsername));

        ArrayList<LinkedHashMap<String, String>> roles = (ArrayList<LinkedHashMap<String, String>>) claimsFromToken.get(claimKeyAuthorities);

        assertEquals(Role.WORKER.getName(), roles.get(0).get("authority"));
    }
}